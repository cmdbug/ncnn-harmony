import { Context } from "@ohos.arkui.UIContext"
import { buffer, JSON } from "@kit.ArkTS"
import { fileIo } from '@kit.CoreFileKit'
import resourceManager from '@ohos.resourceManager';
import { photoAccessHelper } from "@kit.MediaLibraryKit";
import { dataSharePredicates } from "@kit.ArkData";


/**
 * 复制 Rawfile 到沙盒中
 * @param context
 * @param dir 不以 / 开头结尾
 * @param fileName 不以 / 开头，包含扩展名
 */
export function copyRawfileToSanbox(
  context: Context,
  resourceManager: resourceManager.ResourceManager,
  dir: string,
  fileName: string,
  success: () => void
) {
  let boxPath = context.getApplicationContext().filesDir
  let unzipPath = boxPath + "/" + dir
  let zipPath = boxPath + "/" + dir + '/' + fileName

  // 获取rawfile资源文件，转换数据
  let uint8Array: Uint8Array = resourceManager.getRawFileContentSync(dir + '/' + fileName)
  let bf = buffer.from(uint8Array).buffer

  // 创建沙箱目录
  try {
    if (!fileIo.accessSync(unzipPath)) {
      fileIo.mkdirSync(unzipPath)
    }
  } catch (e) {
    console.log("创建沙盒目录异常:" + e)
  }

  try {
    // 打开文件
    const fsOpen = fileIo.openSync(zipPath,
      fileIo.OpenMode.READ_WRITE
        | fileIo.OpenMode.READ_ONLY
        | fileIo.OpenMode.CREATE
        | fileIo.OpenMode.TRUNC)

    // 写入文件(同步会报异常)
    // let destFile = fileIo.writeSync(fsOpen.fd, bf)
    // 关闭文件
    // fileIo.close(destFile)

    // 写入文件
    fileIo.write(fsOpen.fd, bf)
      .then((writeLen: number) => {
        // 关闭文件
        fileIo.close(fsOpen)
        success && success()
      })
  } catch (e) {
    console.log("写入沙盒文件异常:" + e)
    success && success()
  }
}

/**
 * 获取 uri 文件的信息
 * @param path
 * @returns
 */
export async function getUriInfo(path: string) {
  let asset: photoAccessHelper.PhotoAsset | undefined = undefined
  try {
    let phAccessHelper = photoAccessHelper.getPhotoAccessHelper(getContext());
    let predicates: dataSharePredicates.DataSharePredicates = new dataSharePredicates.DataSharePredicates();
    // 配置查询条件，使用PhotoViewPicker选择图片返回的uri进行查询
    predicates.equalTo('uri', path);
    let fetchOption: photoAccessHelper.FetchOptions = {
      fetchColumns: [
        photoAccessHelper.PhotoKeys.WIDTH,
        photoAccessHelper.PhotoKeys.HEIGHT,
        photoAccessHelper.PhotoKeys.SIZE,
        photoAccessHelper.PhotoKeys.TITLE,
        photoAccessHelper.PhotoKeys.DURATION
      ],
      predicates: predicates
    };
    let fetchResult: photoAccessHelper.FetchResult<photoAccessHelper.PhotoAsset> =
      await phAccessHelper.getAssets(fetchOption);
    // 得到uri对应的PhotoAsset对象，读取文件的部分信息
    asset = await fetchResult.getFirstObject();
    // console.info('asset displayName: ', asset.displayName);
    // console.info('asset uri: ', asset.uri);
    // console.info('asset photoType: ', asset.photoType);
    console.info('asset width: ', asset.get(photoAccessHelper.PhotoKeys.WIDTH));
    console.info('asset height: ', asset.get(photoAccessHelper.PhotoKeys.HEIGHT));
    console.info('asset size: ', asset.get(photoAccessHelper.PhotoKeys.SIZE));
    // console.info('asset title: ' + asset.get(photoAccessHelper.PhotoKeys.TITLE));
  } catch (error) {
    console.error('uriGetAssets failed with err: ' + JSON.stringify(error));
  }
  return asset
}


