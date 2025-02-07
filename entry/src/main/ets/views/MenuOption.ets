import { IModelType, modelList } from "../model/ModelList"

@ComponentV2
export struct MenuOption {
  @Consumer('currentModel') currentModel: IModelType = modelList[0]
  @Consumer('currentType') currentType: string = 'CPU'
  @Consumer('currentCore') currentCore: string = '全部'
  @Consumer('currentThread') currentThread: string = '自动'

  checkCurrentModel() {
    const menus: MenuElement[] = []
    for (let index = 0; index < modelList.length; index++) {
      const model = modelList[index]
      const menu: MenuElement = {
        value: model.name,
        action: () => {
          this.currentModel = model
        }
      }
      menus.push(menu)
    }
    return menus
  }

  checkCurrentType() {
    const menus: MenuElement[] = []
    const types: string[] = ['CPU', 'GPU']
    for (let index = 0; index < types.length; index++) {
      const temp = types[index]
      const menu: MenuElement = {
        value: temp,
        action: () => {
          this.currentType = temp
        }
      }
      menus.push(menu)
    }
    return menus
  }

  checkCurrentCore() {
    const menus: MenuElement[] = []
    const cores: string[] = ['全部', '大核', '小核']
    for (let index = 0; index < cores.length; index++) {
      const temp = cores[index]
      const menu: MenuElement = {
        value: temp,
        action: () => {
          this.currentCore = temp
        }
      }
      menus.push(menu)
    }
    return menus
  }

  checkCurrentThread() {
    const menus: MenuElement[] = []
    const threads: string[] = [
      '自动',
      '线程数: 1', '线程数: 2', '线程数: 3', '线程数: 4',
      '线程数: 5', '线程数: 6', '线程数: 7', '线程数: 8'
    ]
    for (let index = 0; index < threads.length; index++) {
      const temp = threads[index]
      const menu: MenuElement = {
        value: temp,
        action: () => {
          this.currentThread = temp
        }
      }
      menus.push(menu)
    }
    return menus
  }

  build() {
    Row({ space: 5 }) {
      Row() {
        Text(this.currentModel.name)
          .fontColor($r('sys.color.white'))
          .layoutWeight(1)
          .padding({ right: 5 })
        Image($r('app.media.icon_arrow_down'))
          .height(15)
          .aspectRatio(1)
          .objectFit(ImageFit.Contain)
      }
      .layoutWeight(2)
      .stylesItem()
      .bindMenu(this.checkCurrentModel())

      Row() {
        Text(this.currentType)
          .fontColor($r('sys.color.white'))
          .padding({ right: 5 })
        Image($r('app.media.icon_arrow_down'))
          .height(15)
          .aspectRatio(1)
          .objectFit(ImageFit.Contain)
      }
      .stylesItem()
      .bindMenu(this.checkCurrentType())

      Row() {
        Text(this.currentCore)
          .fontColor($r('sys.color.white'))
          .padding({ right: 5 })
        Image($r('app.media.icon_arrow_down'))
          .height(15)
          .aspectRatio(1)
          .objectFit(ImageFit.Contain)
      }
      .stylesItem()
      .bindMenu(this.checkCurrentCore())

      Row() {
        Text(this.currentThread)
          .fontColor($r('sys.color.white'))
          .padding({ right: 5 })
        Image($r('app.media.icon_arrow_down'))
          .height(15)
          .aspectRatio(1)
          .objectFit(ImageFit.Contain)
      }
      .stylesItem()
      .bindMenu(this.checkCurrentThread())
    }
    .width('100%')
    .padding({ left: 5, right: 5 })
  }

  @Styles
  stylesItem() {
    .padding({ left: 5, right: 5, top: 7, bottom: 7 })
    .backgroundColor($r('app.color.color_primary'))
    .borderRadius(7)
  }

}
