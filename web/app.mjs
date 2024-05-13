const ONNX_MODEL = 'srcnn.onnx';
const TEST_IMAGE = 'test_img2.jpg';

const SKIP_INFERENCE = false;

const encodeDims = (w, h) => [1, 1, h, w]; // [batch_size, channels, height, width]
const decodeDims = (dims) => [dims[3], dims[2]]; // [w,h]

(function () {
  const { InferenceSession, Tensor } = window.ort;
  const Jimp = window.Jimp;
  const canvasEl = document.getElementById('viewport');
  const statusEl = document.getElementById('text-status');
  const containerModeSelectEl = document.getElementById('img-selector');
  const containerModeResultEl = document.getElementById('img-result');
  const resetResultBtnEl = document.getElementById('img-reset');
  const imgCompareEl = document.getElementById('img-compare');

  let isProcessing = false; // DO NOT CHANGE IN INIT

  hookTestImageBtn();
  hookFileInputZone();
  hookResetBtn(resetResultBtnEl);
  setModeShowResult(false);

  /**
   * https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
   * https://pytorch.org/docs/stable/onnx_torchscript.html#module-torch.onnx
   */
  async function inference(inputData) {
    // ort.env.wasm.simd = false;
    // ort.env.wasm.numThreads = 1;
    // ort.env.logLevel = 'verbose'; // ?
    // ort.env.debug = true; // ?
    ort.env.trace = true;

    const [w, h] = decodeDims(inputData.dims);
    // https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
    const session = await InferenceSession.create(ONNX_MODEL, {
      executionProviders: [
        'cpu', // OK, 1.632s
        // 'wasm', // OK, 1.629s
        // 'webgpu', // requires https. (May 2024) webgpu backend still has too many errors
        // 'webgl', // ERROR: input tensor[0] check failed: expected shape '[,,,]' but got [1,1,1280,1994]
      ],
      // graphOptimizationLevel: 'all',
      // freeDimensionOverrides: {
      // in_w: w,
      // out_w: w,
      // in_h: h,
      // out_h: h,
      // },
    });
    console.log('Inference session created', session);

    // inputs
    const feeds = {};
    feeds[session.inputNames[0]] = inputData;

    // run
    const start = new Date();
    const outputData = await session.run(feeds, {
      // logSeverityLevel: 0, // ?
      // logVerbosityLevel: 0, // ?
    });

    // end - timer
    const end = new Date();
    const inferenceTime = (end.getTime() - start.getTime()) / 1000;

    // result
    const output = outputData[session.outputNames[0]];
    console.log('output: ', output);
    return [output, inferenceTime];
  }

  async function inferenceImage(imageToInfer) {
    if (isProcessing) return;
    setProcessing(true);

    try {
      console.log('Input img:', imageToInfer);
      const [orgImage, imageTensor] = await getImageTensorFromPath(
        imageToInfer
      );
      console.log('Input tensor:', imageTensor);

      if (!SKIP_INFERENCE) {
        const [result, inferenceTime] = await inference(imageTensor);
        console.log(`Inference speed: ${inferenceTime} seconds`);
        setStatus(
          `Inference speed: ${inferenceTime}s.`,
          `Size: ${imageTensor.dims[2]}x${imageTensor.dims[3]} px`
        );

        await drawResult(canvasEl, result, orgImage);
      } else {
        await drawResult(canvasEl, imageTensor, orgImage);
        setStatus(`Drawn input image`);
      }
    } catch (e) {
      setStatus('ERROR:', e.message);
      throw e;
    } finally {
      setProcessing(false);
    }
  }

  function setProcessing(nextIsProcessing) {
    isProcessing = nextIsProcessing;
    if (isProcessing) {
      setStatus('Processing..');
      setModeShowResult(true);
      hideEl(canvasEl);
      hideEl(imgCompareEl);
      hideEl(resetResultBtnEl);
    } else {
      showEl(resetResultBtnEl);
    }
  }

  ///////////////
  /// IMAGE HANDLING

  async function getImageTensorFromPath(path) {
    var [image, width, height] = await loadImagefromPath(path);
    var imageTensor = imageDataToTensor(image, width, height);
    return [image, imageTensor];
  }

  async function loadImagefromPath(path) {
    const getImgSize = (imageObj) => {
      if (imageObj._exif) return imageObj._exif.imageSize;
      return imageObj.bitmap;
    };

    const imageData = await Jimp.read(path).then((imageObj) => {
      console.log('Upscaling:', imageObj);
      // console.log('w-h', { width, height });
      const { width, height } = getImgSize(imageObj);
      const upscaledWidth = width * 2;
      const upscaledHeight = height * 2;
      // const upscaledHeight = width * 2; // test square images
      console.log('Size:', {
        width,
        height,
        upscaledWidth,
        upscaledHeight,
      });
      const resizedImage = imageObj.resize(upscaledWidth, upscaledHeight);
      return [resizedImage, upscaledWidth, upscaledHeight];
    });

    return imageData;
  }

  function imageDataToTensor(image, width, height) {
    const imageBufferData = image.bitmap.data;

    const outSize = width * height;
    // const dims = [1, 1, width, height];
    // const dims = [1, 1, height, width];
    const dims = encodeDims(width, height);

    const float32Data = new Float32Array(outSize);

    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        // for (let i = 0; i < outSize; i += 1) {
        // const i = x * height + y;
        const i = y * width + x;
        const r = imageBufferData[i * 4];
        const g = imageBufferData[i * 4 + 1];
        const b = imageBufferData[i * 4 + 2];
        const greyscale = 0.2989 * r + 0.587 * g + 0.114 * b;
        float32Data[i] = greyscale / 255.0; // convert to float
        // float32Data[i] = 1.0;
      }
    }

    const inputTensor = new Tensor('float32', float32Data, dims);
    return inputTensor;
  }

  ///////////////
  /// UI

  function setModeShowResult(isResult) {
    if (isResult) {
      console.log('MODE: result');
      hideEl(containerModeSelectEl);
      showEl(containerModeResultEl);
    } else {
      console.log('MODE: img select');
      showEl(containerModeSelectEl);
      hideEl(containerModeResultEl);
    }
  }

  function hookFileInputZone() {
    const fileInputZoneEl = document.getElementById('img-dropzone');
    const fileInputEl = document.querySelector('#img-dropzone input');

    const setDragClass = (v) => setClass(fileInputZoneEl, 'drop', v);

    fileInputEl.onchange = async function (e) {
      setDragClass(false);
      const file = this.files[0];

      const accept = this.accept ? this.accept.split(/, ?/) : undefined;
      if (!accept.includes(file.type)) {
        alert(
          `File type not allowed. Expected: '${this.accept}'. Got: ${file.type}`
        );
        return;
      }

      const arrayBuffer = await file.arrayBuffer();
      inferenceImage(arrayBuffer);
    };

    fileInputZoneEl.ondragenter = (e) => {
      // console.log('ondragenter');
      e.preventDefault();
      setDragClass(true);
    };
    fileInputZoneEl.ondragleave = (e) => {
      // console.log('ondragleave');
      e.preventDefault();
      setDragClass(false);
    };
  }

  function hookTestImageBtn() {
    const btnEl = document.getElementById('use-test-image-btn');
    btnEl.onclick = () => {
      inferenceImage(TEST_IMAGE);
    };
  }

  function hookResetBtn(el) {
    el.onclick = () => setModeShowResult(false);
  }

  async function drawResult(canvasEl, result, orgImage) {
    console.log('Drawing to canvas: ', result);
    console.log('Original image: ', orgImage);
    const data = result.cpuData;
    const [maxW, maxH] = decodeDims(result.dims);
    const orgImgData = orgImage.bitmap.data;

    showEl(canvasEl);
    canvasEl.width = maxW;
    canvasEl.height = maxH;

    const ctx = canvasEl.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height);
    const pixels = imgData.data;

    for (let x = 0; x < maxW; x++) {
      for (let y = 0; y < maxH; y++) {
        // org image
        const orgImgIdx = x * maxH + y;
        const r = orgImgData[orgImgIdx * 4];
        const g = orgImgData[orgImgIdx * 4 + 1];
        const b = orgImgData[orgImgIdx * 4 + 2];

        // sample luma
        const idx = x * maxH + y;
        const new_luma = Math.floor(data[idx] * 255);
        // show grayscale - debug
        // pixels[idx * 4 + 0] = new_luma;
        // pixels[idx * 4 + 1] = new_luma;
        // pixels[idx * 4 + 2] = new_luma;
        // pixels[idx * 4 + 3] = 255;

        // combine: swap luma
        // see srcnn.py for docs
        const pr = 0.5 * r - 0.419 * g - 0.081 * b;
        const pb = -0.169 * r - 0.331 * g + 0.5 * b;
        pixels[idx * 4 + 0] = new_luma + 0.0 * pb + 1.402 * pr;
        pixels[idx * 4 + 1] = new_luma - 0.344 * pb - 0.714 * pr;
        pixels[idx * 4 + 2] = new_luma + 1.772 * pb + 0.0 * pr;
        pixels[idx * 4 + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    ///
    const imgBefore = document.getElementById('img-compare-before');
    imgBefore.src = await orgImage.getBase64Async(Jimp.AUTO);
    // imgBefore.width = maxW;
    // imgBefore.height = maxH;
    const imgAfter = document.getElementById('img-compare-after');
    imgAfter.src = canvasEl.toDataURL();
    // imgAfter.width = maxW;
    // imgAfter.height = maxH;

    const ImageCompare = window.ImageCompare;
    const viewer = new ImageCompare(imgCompareEl).mount();

    hideEl(canvasEl);
    showEl(imgCompareEl);
  }

  function setStatus(...texts) {
    statusEl.textContent = texts.join(' ');
  }
})();

function setClass(el, clazz, state) {
  if (state) {
    el.classList.add(clazz);
  } else {
    el.classList.remove(clazz);
  }
}

function showEl(el) {
  setClass(el, 'mode-hidden', false);
}

function hideEl(el) {
  setClass(el, 'mode-hidden', true);
}
