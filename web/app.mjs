const ONNX_MODEL = 'srcnn.onnx';
const TEST_IMAGE = 'test_img.jpg';
const TEST_IMAGE_DIMS = [1, 1000, 1000];

(function () {
  const { InferenceSession, Tensor } = window.ort;
  const Jimp = window.Jimp;

  /**
   * https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
   */
  async function infere(inputData) {
    // https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
    const session = await InferenceSession.create(ONNX_MODEL, {
      executionProviders: [
        // 'webgpu',
        // 'webgl',
      ],
      graphOptimizationLevel: 'all',
    });
    console.log('Inference session created');
    console.log(session);

    // inputs
    const feeds = {};
    feeds[session.inputNames[0]] = inputData;

    // run
    const start = new Date();
    const outputData = await session.run(feeds);

    // end - timer
    const end = new Date();
    const inferenceTime = (end.getTime() - start.getTime()) / 1000;

    // result
    const output = outputData[session.outputNames[0]];
    console.log('output: ', output);
    return [output, inferenceTime];
  }

  async function infereTestImage() {
    let imageTensor = await getImageTensorFromPath(TEST_IMAGE);
    // console.log(imageTensor);
    imageTensor = imageTensor.reshape([1, ...imageTensor.dims]);
    const [result, inferenceTime] = await infere(imageTensor);
    console.log(`Inference speed: ${inferenceTime} seconds`);
    drawResult(result);
    // drawResult(imageTensor);
  }

  document.getElementById('test-infere-btn').onclick = infereTestImage;

  function drawResult(result) {
    console.log('Drawing to canvas: ', result);
    const data = result.cpuData;
    const maxW = result.dims[2];
    const maxH = result.dims[3];

    const canvas = document.getElementById('viewport');
    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imgData.data;

    for (let x = 0; x < maxW; x++) {
      for (let y = 0; y < maxH; y++) {
        const idx = y * maxW + x;
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

  ///////////////
  /// IMAGE HANDLING

  async function getImageTensorFromPath(path, dims = TEST_IMAGE_DIMS) {
    var image = await loadImagefromPath(path, dims[1], dims[2]);
    var imageTensor = imageDataToTensor(image, dims);
    return imageTensor;
  }

  async function loadImagefromPath(path, width, height) {
    var imageData = await Jimp.read(path).then((imageBuffer) => {
      return imageBuffer.resize(width, height);
    });

    return imageData;
  }

  function imageDataToTensor(image, dims) {
    var imageBufferData = image.bitmap.data;

    const outSize = dims[0] * dims[1] * dims[2];
    const float32Data = new Float32Array(outSize);
    for (let i = 0; i < outSize; i += 1) {
      const r = imageBufferData[i * 4];
      const g = imageBufferData[i * 4 + 1];
      const b = imageBufferData[i * 4 + 2];
      const greyscale = 0.2989 * r + 0.587 * g + 0.114 * b;
      float32Data[i] = greyscale / 255.0; // convert to float
      // float32Data[i] = 1.0;
    }

    const inputTensor = new Tensor('float32', float32Data, dims);
    return inputTensor;
  }
})();
