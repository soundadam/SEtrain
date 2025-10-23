#!/bin/bash

# 南京大学云盘上传脚本（并行优化版）
# 用法: ./upload_to_nju.sh <文件或文件夹路径>

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <文件或文件夹路径>"
    echo "示例: $0 /path/to/file.yaml"
    echo "      $0 /path/to/folder"
    exit 1
fi

PATH_TO_UPLOAD="$1"

# 检查路径是否存在
if [ ! -e "$PATH_TO_UPLOAD" ]; then
    echo "错误: 路径不存在: $PATH_TO_UPLOAD"
    exit 1
fi

# 配置变量
UPLOAD_PAGE_URL=https://box.nju.edu.cn/u/d/1de5d011e7134ab48ddc/
UPLOAD_TOKEN="1de5d011e7134ab48ddc"
UPLOAD_DIR="/datasets/"
COOKIE_FILE_PREFIX="/tmp/nju_cookies"

# 并行配置
MAX_PARALLEL_UPLOADS=${MAX_PARALLEL_UPLOADS:-4}

# 文件大小限制（15GB = 15 * 1024 * 1024 * 1024 bytes）
MAX_FILE_SIZE=$((15 * 1024 * 1024 * 1024))

# 计数器（需要共享文件来跨进程统计）
STATS_DIR="/tmp/nju_upload_stats_$$"
mkdir -p "$STATS_DIR"
TOTAL_FILES_FILE="$STATS_DIR/total"
SUCCESS_COUNT_FILE="$STATS_DIR/success"
FAILED_COUNT_FILE="$STATS_DIR/failed"
SKIPPED_COUNT_FILE="$STATS_DIR/skipped"

echo "0" > "$TOTAL_FILES_FILE"
echo "0" > "$SUCCESS_COUNT_FILE"
echo "0" > "$FAILED_COUNT_FILE"
echo "0" > "$SKIPPED_COUNT_FILE"

# 获取 CSRF token（每个进程使用独立的 cookie 文件）
get_csrf_token() {
    local COOKIE_FILE="${1:-${COOKIE_FILE_PREFIX}_$$.txt}"
    curl -s -c "$COOKIE_FILE" -b "$COOKIE_FILE" "$UPLOAD_PAGE_URL" > /dev/null
    if [ $? -ne 0 ]; then
        echo "错误: 无法获取 CSRF token"
        return 1
    fi
    echo "$COOKIE_FILE"
    return 0
}

# 获取上传链接
get_upload_link() {
    local COOKIE_FILE="$1"
    local UPLOAD_LINK_JSON=$(curl -s -b "$COOKIE_FILE" \
        -H "Referer: $UPLOAD_PAGE_URL" \
        "https://box.nju.edu.cn/api/v2.1/upload-links/$UPLOAD_TOKEN/upload/")

    if [ $? -ne 0 ]; then
        echo "错误: 无法获取上传链接"
        return 1
    fi

    local LINK=$(echo "$UPLOAD_LINK_JSON" | grep -o '"upload_link":"[^"]*"' | sed 's/"upload_link":"//;s/"//')

    if [ -z "$LINK" ]; then
        echo "错误: 无法解析上传链接"
        return 1
    fi

    echo "$LINK"
    return 0
}

# 增加计数器（线程安全）
increment_counter() {
    local counter_file="$1"
    local lock_file="${counter_file}.lock"

    # 使用文件锁确保原子操作
    (
        flock -x 200
        local count=$(cat "$counter_file")
        echo $((count + 1)) > "$counter_file"
    ) 200>"$lock_file"
}

# 格式化文件大小
format_size() {
    local size=$1
    if [ $size -lt 1024 ]; then
        echo "${size}B"
    elif [ $size -lt $((1024 * 1024)) ]; then
        echo "$((size / 1024))KB"
    elif [ $size -lt $((1024 * 1024 * 1024)) ]; then
        echo "$((size / 1024 / 1024))MB"
    else
        echo "$((size / 1024 / 1024 / 1024))GB"
    fi
}

# 检查文件大小
check_file_size() {
    local file_path="$1"
    local file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null)

    if [ -z "$file_size" ]; then
        echo "0"
        return 1
    fi

    echo "$file_size"
    return 0
}

# 上传单个文件
upload_file() {
    local FILE_PATH="$1"
    local TARGET_DIR="$2"
    local COOKIE_FILE="$3"

    increment_counter "$TOTAL_FILES_FILE"

    # 检查文件大小
    local file_size=$(check_file_size "$FILE_PATH")
    local size_str=$(format_size $file_size)

    if [ $file_size -gt $MAX_FILE_SIZE ]; then
        echo "[跳过] $FILE_PATH (大小: $size_str > 15GB)"
        increment_counter "$SKIPPED_COUNT_FILE"
        return 2
    fi

    echo "[上传] $FILE_PATH (大小: $size_str) -> $TARGET_DIR"

    # 获取上传链接
    local UPLOAD_LINK=$(get_upload_link "$COOKIE_FILE")
    if [ $? -ne 0 ]; then
        echo "  [失败] 无法获取上传链接"
        increment_counter "$FAILED_COUNT_FILE"
        return 1
    fi

    # 上传文件
    local UPLOAD_RESULT=$(curl -s -F "file=@$FILE_PATH" \
        -F "parent_dir=$TARGET_DIR" \
        "$UPLOAD_LINK")

    if [ $? -ne 0 ]; then
        echo "  [失败] 文件上传失败"
        increment_counter "$FAILED_COUNT_FILE"
        return 1
    fi

    echo "  [成功] $UPLOAD_RESULT"
    increment_counter "$SUCCESS_COUNT_FILE"
    return 0
}

# 收集所有需要上传的文件
collect_files() {
    local FOLDER_PATH="$1"
    local BASE_PATH="$2"
    local FILES_LIST="$3"

    local RELATIVE_PATH="${FOLDER_PATH#$BASE_PATH}"
    RELATIVE_PATH="${RELATIVE_PATH#/}"

    local TARGET_DIR="$UPLOAD_DIR$RELATIVE_PATH"
    [ -n "$RELATIVE_PATH" ] && TARGET_DIR="$TARGET_DIR/"

    # 收集当前文件夹中的所有文件
    while IFS= read -r -d '' FILE; do
        echo "$FILE|$TARGET_DIR" >> "$FILES_LIST"
    done < <(find "$FOLDER_PATH" -maxdepth 1 -type f -print0)

    # 递归处理子文件夹
    while IFS= read -r -d '' SUBFOLDER; do
        collect_files "$SUBFOLDER" "$BASE_PATH" "$FILES_LIST"
    done < <(find "$FOLDER_PATH" -maxdepth 1 -type d ! -path "$FOLDER_PATH" -print0)
}

# 并行上传处理器
parallel_upload_worker() {
    local file_path="$1"
    local target_dir="$2"

    # 每个进程使用独立的 cookie 文件
    local worker_cookie="${COOKIE_FILE_PREFIX}_worker_$$.txt"

    # 获取独立的 CSRF token
    local cookie_file=$(get_csrf_token "$worker_cookie")
    if [ $? -ne 0 ]; then
        echo "  [失败] 无法初始化 worker cookie"
        increment_counter "$FAILED_COUNT_FILE"
        return 1
    fi

    # 上传文件
    upload_file "$file_path" "$target_dir" "$cookie_file"
    local result=$?

    # 清理 cookie 文件
    rm -f "$cookie_file"

    return $result
}

# 导出函数供子进程使用
export -f get_csrf_token
export -f get_upload_link
export -f upload_file
export -f increment_counter
export -f format_size
export -f check_file_size
export -f parallel_upload_worker
export UPLOAD_PAGE_URL
export UPLOAD_TOKEN
export UPLOAD_DIR
export COOKIE_FILE_PREFIX
export MAX_FILE_SIZE
export STATS_DIR
export TOTAL_FILES_FILE
export SUCCESS_COUNT_FILE
export FAILED_COUNT_FILE
export SKIPPED_COUNT_FILE

# 主程序
echo "=========================================="
echo "南京大学云盘上传工具（并行优化版）"
echo "=========================================="
echo "目标云盘: $UPLOAD_PAGE_URL"
echo "最大文件大小: 15GB"
echo "并行上传数: $MAX_PARALLEL_UPLOADS"
echo ""

# 获取 CSRF token（初始化）
echo "初始化连接..."
MAIN_COOKIE_FILE=$(get_csrf_token "${COOKIE_FILE_PREFIX}_main.txt")
if [ $? -ne 0 ]; then
    rm -rf "$STATS_DIR"
    exit 1
fi
echo "连接成功!"
echo ""

# 判断是文件还是文件夹
if [ -f "$PATH_TO_UPLOAD" ]; then
    # 单个文件
    echo "上传模式: 单文件"
    upload_file "$PATH_TO_UPLOAD" "$UPLOAD_DIR" "$MAIN_COOKIE_FILE"

elif [ -d "$PATH_TO_UPLOAD" ]; then
    # 文件夹 - 并行上传
    echo "上传模式: 文件夹（并行）"
    echo "扫描路径: $PATH_TO_UPLOAD"
    echo ""

    # 获取绝对路径
    BASE_PATH=$(cd "$PATH_TO_UPLOAD" && pwd)

    # 收集所有文件
    FILES_LIST="$STATS_DIR/files_list.txt"
    echo "正在收集文件列表..."
    collect_files "$BASE_PATH" "$BASE_PATH" "$FILES_LIST"

    TOTAL_FILE_COUNT=$(wc -l < "$FILES_LIST" 2>/dev/null || echo "0")
    echo "找到 $TOTAL_FILE_COUNT 个文件"
    echo ""

    if [ $TOTAL_FILE_COUNT -eq 0 ]; then
        echo "没有找到需要上传的文件"
    else
        echo "开始并行上传..."
        echo ""

        # 并行上传
        active_jobs=0
        pids=()
        while IFS='|' read -r file_path target_dir; do
            # 控制并发数：等待有空闲位置
            while [ $active_jobs -ge $MAX_PARALLEL_UPLOADS ]; do
                # 检查已完成的任务
                for i in "${!pids[@]}"; do
                    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                        # 任务已完成，等待以回收资源
                        wait "${pids[$i]}" 2>/dev/null
                        unset 'pids[$i]'
                        ((active_jobs--))
                    fi
                done
                # 重新构建 pids 数组，移除空元素
                pids=("${pids[@]}")

                # 如果还是满的，稍作等待
                if [ $active_jobs -ge $MAX_PARALLEL_UPLOADS ]; then
                    sleep 0.2
                fi
            done

            # 启动后台上传任务
            parallel_upload_worker "$file_path" "$target_dir" &
            pids+=($!)
            ((active_jobs++))
        done < "$FILES_LIST"

        # 等待所有任务完成
        echo ""
        echo "等待所有上传任务完成..."
        for pid in "${pids[@]}"; do
            [ -n "$pid" ] && wait "$pid" 2>/dev/null
        done
    fi
fi

# 清理所有 cookie 文件
rm -f ${COOKIE_FILE_PREFIX}*.txt

# 读取统计数据
TOTAL_FILES=$(cat "$TOTAL_FILES_FILE")
SUCCESS_COUNT=$(cat "$SUCCESS_COUNT_FILE")
FAILED_COUNT=$(cat "$FAILED_COUNT_FILE")
SKIPPED_COUNT=$(cat "$SKIPPED_COUNT_FILE")

# 清理统计目录
rm -rf "$STATS_DIR"

# 显示统计
echo ""
echo "=========================================="
echo "上传完成!"
echo "=========================================="
echo "总文件数: $TOTAL_FILES"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAILED_COUNT"
echo "跳过(>15GB): $SKIPPED_COUNT"
echo "=========================================="

exit 0
