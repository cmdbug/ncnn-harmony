import { IOptionType } from "../types/Types"


@ComponentV2
export struct OptionDialogContent {
  @Param @Require option: IOptionType
  @Param @Require action: (option: IOptionType) => void = (option: IOptionType) => {
  }

  build() {
    Column({ space: 10 }) {
      Text('参数配置')
        .width('100%')
        .fontSize(18)
        .textAlign(TextAlign.Center)
      List({ space: 10 }) {
        ListItem() {
          Row() {
            Checkbox({ name: 'Mempool' })
              .select(this.option.mempool)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.mempool = value
              })
            Text('Mempool')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'Winograd' })
              .select(this.option.winograd)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.winograd = value
              })
            Text('Winograd')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'SGEMM' })
              .select(this.option.sgemm)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.sgemm = value
              })
            Text('SGEMM')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'Pack4' })
              .select(this.option.pack4)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.pack4 = value
              })
            Text('Pack4')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'BF16 Storage' })
              .select(this.option.bf16Storage)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.bf16Storage = value
              })
            Text('BF16 Storage')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'FP16 Packed' })
              .select(this.option.fp16Packed)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.fp16Packed = value
              })
            Text('FP16 Packed')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'FP16 Storage' })
              .select(this.option.fp16Storage)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.fp16Storage = value
              })
            Text('FP16 Storage')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'FP16 Arithmetic' })
              .select(this.option.fp16Arithmetic)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.fp16Arithmetic = value
              })
            Text('FP16 Arithmetic')
          }
        }

        ListItem() {
          Row() {
            Checkbox({ name: 'Light Mode' })
              .select(this.option.lightMode)
              .selectedColor($r('app.color.color_primary'))
              .shape(CheckBoxShape.ROUNDED_SQUARE)
              .onChange((value: boolean) => {
                this.option.lightMode = value
              })
            Text('Light Mode')
          }
        }

        ListItem() {
          Row() {
            Blank()
            Button('修改')
              .backgroundColor($r('app.color.color_primary'))
              .onClick(() => {
                console.log('点击修改')
                this.action(this.option)
              })
          }
          .width('100%')
        }
      }
      .width('100%')
      .height('100%')
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Start)
    .alignItems(HorizontalAlign.Start)
    .padding(15)
  }
}
