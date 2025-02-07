export interface IOptionType {
  mempool: boolean
  winograd: boolean
  sgemm: boolean
  pack4: boolean
  bf16Storage: boolean
  fp16Packed: boolean
  fp16Storage: boolean
  fp16Arithmetic: boolean
  lightMode: boolean
}

export interface IConfigType {
  isGPU: boolean
  core: number // 0全部 1大核 2小核
  thread: number // 0自动 1...8线程数
}






