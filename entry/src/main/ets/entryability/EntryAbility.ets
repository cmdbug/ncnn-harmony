import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { BOTTOM_HEIGHT, TOP_HEIGHT } from '../utils/ConstantKey';
import { AppUtil } from '@pura/harmony-utils';
import { globalCrashCatch } from '../jsexception/JSGlobalException';

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onCreate');

    globalCrashCatch()
  }

  onDestroy(): void {
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    // Main window is created, set main page for this ability
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageCreate');

    windowStage.loadContent('pages/Main', (err) => {
      if (err.code) {
        hilog.error(0x0000, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err) ?? '');
        return;
      }
      hilog.info(0x0000, 'testTag', 'Succeeded in loading the content.');
    });

    // 设置全屏
    let windowClass = windowStage.getMainWindowSync()
    try {
      windowClass.setWindowLayoutFullScreen(true)
    } catch (e) {
    }
    // 安全区域
    this.setOrCreateSafeAreaView(windowClass)
    // 状态栏为白色
    windowClass.setWindowSystemBarProperties({
      // 状态栏文字颜色
      statusBarContentColor: '#FFFFFF'
    })

    AppUtil.init(this.context)
  }

  onWindowStageDestroy(): void {
    // Main window is destroyed, release UI related resources
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    // Ability has brought to foreground
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    // Ability has back to background
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onBackground');
  }

  /**
   * 设置或创建安全区域
   * @param windowClass
   */
  setOrCreateSafeAreaView(windowClass: window.Window) {
    // 安全区域
    let topHeight = px2vp(windowClass.getWindowAvoidArea(window.AvoidAreaType.TYPE_SYSTEM)
      .topRect
      .height) // 获取状态栏height
    let bottomHeight = px2vp(windowClass.getWindowAvoidArea(window.AvoidAreaType.TYPE_NAVIGATION_INDICATOR)
      .bottomRect
      .height) // 获取导航栏height
    // console.log("状态栏:" + topHeight, "底部:" + bottomHeight)
    AppStorage.setOrCreate(TOP_HEIGHT, topHeight)
    AppStorage.setOrCreate(BOTTOM_HEIGHT, bottomHeight)
    // 安全区域发生了变化
    windowClass.on('avoidAreaChange', (data) => {
      if (data.type === window.AvoidAreaType.TYPE_SYSTEM) {
        let topRectHeight = px2vp(data.area.topRect.height)
        AppStorage.setOrCreate(TOP_HEIGHT, topRectHeight)
      } else if (data.type == window.AvoidAreaType.TYPE_NAVIGATION_INDICATOR) {
        let bottomRectHeight = px2vp(data.area.bottomRect.height)
        AppStorage.setOrCreate(BOTTOM_HEIGHT, bottomRectHeight)
      }
    })
  }
}
