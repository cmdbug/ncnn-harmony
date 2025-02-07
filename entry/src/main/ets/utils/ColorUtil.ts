// 根据字符串生成颜色
export function stringToColor(str: string): string {
  // 将输入字符串转换为哈希值
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0; // 强制转换为 32 位整数
  }

  // 使用哈希值生成颜色，确保颜色为 HEX 格式
  const color = (hash & 0x00FFFFFF).toString(16).toUpperCase(); // 取哈希值的低 24 位
  return `#${'000000'.substring(0, 6 - color.length) + color}`; // 保证颜色是 6 位长，不足前面补零
}