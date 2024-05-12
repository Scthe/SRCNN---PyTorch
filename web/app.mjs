const ONNX_MODEL = 'srcnn.onnx';
const IMAGES = ['test_img.jpg', 'test_img2.jpg'].map((e) => `imgs/${e}`);

// TODO reexport the onnx with correct dim names
// TODO color mode
// TODO image upload
// TODO finish README

const SKIP_INFERENCE = false;

const encodeDims = (w, h) => [1, 1, h, w]; // [batch_size, channels, height, width]
const decodeDims = (dims) => [dims[3], dims[2]]; // [w,h]

(function () {
  const { InferenceSession, Tensor } = window.ort;
  const Jimp = window.Jimp;
  const canvasEl = document.getElementById('viewport');
  const statusEl = document.getElementById('text-status');

  let imageToInfer = IMAGES[0];
  let isProcessing = false; // DO NOT CHANGE IN INIT

  document.getElementById('test-infere-btn').onclick = () =>
    inferenceTestImage();
  createImgSelector();

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
        // 'webgpu', // Error: no available backend found. ERR: [webgpu] RuntimeError: indirect call to null
        // 'webgl', // ERROR: input tensor[0] check failed: expected shape '[,,,]' but got [1,1,1280,1994]
        'wasm', // OK
        // 'cpu', // OK
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

  async function inferenceTestImage() {
    if (isProcessing) return;
    setProcessing(true);

    try {
      let imageTensor = await getImageTensorFromPath(imageToInfer);
      console.log('Input tensor:', imageTensor);

      if (!SKIP_INFERENCE) {
        const [result, inferenceTime] = await inference(imageTensor);
        console.log(`Inference speed: ${inferenceTime} seconds`);
        setStatus(
          `Inference speed: ${inferenceTime}s.`,
          `Size: ${imageTensor.dims[2]}x${imageTensor.dims[3]} px`
        );

        drawResult(result);
      } else {
        drawResult(imageTensor);
        setStatus(`Drawn input image`);
      }
    } catch (e) {
      setStatus('ERROR:' + e.message);
      throw e;
    } finally {
      setProcessing(false);
    }
  }

  function drawResult(result) {
    console.log('Drawing to canvas: ', result);
    const data = result.cpuData;
    const [maxW, maxH] = decodeDims(result.dims);

    canvasEl.width = maxW;
    canvasEl.height = maxH;

    const ctx = canvasEl.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height);
    const pixels = imgData.data;

    for (let x = 0; x < maxW; x++) {
      for (let y = 0; y < maxH; y++) {
        // const idx = y * maxW + x;
        const idx = x * maxH + y;
        const grayscale = Math.floor(data[idx] * 255);
        // if (y === 200) console.log({ grayscale });

        pixels[idx * 4 + 0] = grayscale;
        pixels[idx * 4 + 1] = grayscale;
        pixels[idx * 4 + 2] = grayscale;
        pixels[idx * 4 + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }

  function createImgSelector() {
    const container = document.getElementById('img-selector');

    IMAGES.forEach((imgName) => {
      const el = document.createElement('img');
      el.src = imgName;
      el.className = 'preview-img';
      el.onclick = () => {
        console.log('img:', imgName);
        imageToInfer = imgName;
        inferenceTestImage();
      };
      container.appendChild(el);
    });
  }

  function setProcessing(nextIsProcessing) {
    console.log(`--- PROCESSING: ${nextIsProcessing} ---`);
    isProcessing = nextIsProcessing;
    if (isProcessing) {
      setStatus('Processing..');
      canvasEl.style.opacity = 0;
    } else {
      canvasEl.style.opacity = 1;
    }
  }

  function setStatus(...texts) {
    statusEl.textContent = texts.join(' ');
  }

  ///////////////
  /// IMAGE HANDLING

  async function getImageTensorFromPath(path) {
    var [image, width, height] = await loadImagefromPath(path);
    var imageTensor = imageDataToTensor(image, width, height);
    return imageTensor;
  }

  async function loadImagefromPath(path) {
    const imageData = await Jimp.read(path).then((imageObj) => {
      console.log('Upscaling:', imageObj);
      // console.log('w-h', { width, height });
      const { width, height } = imageObj._exif.imageSize;
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
})();
