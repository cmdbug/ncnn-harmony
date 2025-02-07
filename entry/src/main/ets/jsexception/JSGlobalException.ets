import { errorManager } from "@kit.AbilityKit";
import { promptAction } from "@kit.ArkUI";
import { BusinessError } from "@kit.BasicServicesKit";

/**
 * 全局 js 未处理异常捕获
 */
export function globalCrashCatch() {
  let observer1: errorManager.ErrorObserver = {
    onUnhandledException(errorMsg) {
      console.log('onUnhandledException, errorMsg: ', errorMsg);
    },
    onException(errorObj) {
      console.log('onException, name: ', errorObj.name);
      console.log('onException, message: ', errorObj.message);
      if (typeof (errorObj.stack) === 'string') {
        console.log('onException, stack: ', errorObj.stack);
      }
      promptAction.showToast({
        message: "警告：应用出现了未知异常"
      })
    }
  };
  let observerId = -1;
  try {
    // 注册错误观测器。注册后可以捕获到应用产生的js crash，应用崩溃时进程不会退出。
    observerId = errorManager.on('error', observer1);
  } catch (paramError) {
    let code = (paramError as BusinessError).code;
    let message = (paramError as BusinessError).message;
    console.error(`error: ${code}, ${message}`);
  }

  let observer2: errorManager.UnhandledRejectionObserver = (reason: Error, promise: Promise<void>) => {
    if (promise === promise1) {
      console.log("promise1 is rejected");
    }
    console.log("reason.name: ", reason.name);
    console.log("reason.message: ", reason.message);
    if (reason.stack) {
      console.log("reason.stack: ", reason.stack);
    }
    promptAction.showToast({
      message: "警告：应用出现了未知错误"
    })
  };

  // 注册被拒绝promise监听器。注册后可以捕获到当前线程中未被捕获到的promise rejection。
  errorManager.on("unhandledRejection", observer2);

  let promise1 = new Promise<void>(() => {
  }).then(() => {
    throw new Error("uncaught error");
  });
}
