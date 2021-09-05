/** Global Constants */
const MODEL_PATH = 'tfjs/Compiled/model.json';
const IMAGE_RESIZE = 28;
const IMAGE_PADDING_VALUE = 2;
const INPUT_SIZE = 32
// asserts (INPUT_SIZE === IMAGE_SIZE + IMAGE_PADDING_VALUE*2);
const INITIAL_MESSAGE = 'Draw any digit between <strong>0</strong> to <strong>9</strong>';
const MAX_CANVAS_SIZE = 400;
const MAX_CTX_SIZE = 22;
const CANVAS_RESIZE_SUBTRACT_VALUE = 30;
const CTX_SCALE_DIVISOR_VALUE = MAX_CANVAS_SIZE / MAX_CTX_SIZE;
const TIME_TO_WAIT_BEFORE_PREDICT_ON_STOP_DRAWING = 1300;
const TIME_TO_WAIT_BEFORE_PREDICT_ON_MOUSE_OUT = 1500;
const TIME_TO_WAIT_BEFORE_PREDICT_THE_IMAGE = 333;
const PAGE_RESIZE_ADD_VALUE = 200;
const NumberToWord = {
    0: 'Zero',
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine'
}

/** Global Variables */
let model;
let isModelLoaded = false;
let lastPosition = { x: 0, y: 0 };
let drawing = false;
let stopPrediction = false;
let haveAlreadyPredicted = false;
let canvas;
let ctx;


function min(...args)
{
    if (args.length < 2)
        throw Error('At least 2 elements are required for calculating the minimum!');
    let minimun = args[0];
    for (let i = 1 ; i < args.length ; ++i)
        minimun = minimun > args[i] ? args[i] : minimun;
    return minimun;
}


function max(...args)
{
    if (args.length < 2)
        throw Error('At least 2 elements are required for calculating the maximum!');
    let maximum = args[0];
    for (let i = 1 ; i < args.length ; ++i)
        maximum = maximum < args[i] ? args[i] : maximum;
    return maximum;
}


function printOutput(msg = INITIAL_MESSAGE)
{
    const out = document.getElementById('predict-output'); 
    out.innerHTML = msg;
}


function resizePage()
{
    const main = document.getElementsByTagName('html')[0];
    const clear_height = 45;
    const output_height = 40;

    const summed_height = clear_height + output_height + getCanvasSize() + PAGE_RESIZE_ADD_VALUE;

    main.style.height = max(window.innerHeight, summed_height)+'px';
}


function getCanvasSize()
{
    // Prevents some errors when resizing the canvas
    const window_width = window.outerWidth > 0  ?
        min(window.innerWidth, window.outerWidth) : window.innerWidth;

    const width = window_width > (MAX_CANVAS_SIZE + CANVAS_RESIZE_SUBTRACT_VALUE) ?
        MAX_CANVAS_SIZE : (window_width - CANVAS_RESIZE_SUBTRACT_VALUE);

    return width;
}


function getCtxSize()
{
    return getCanvasSize() / CTX_SCALE_DIVISOR_VALUE;
}


function sleep(milisecs)
{
    // Stops the execution by 'milisecs' miliseconds.
    return new Promise(resolve => setTimeout(resolve, milisecs));
}


function resizeCanvas()
{
    // Get the canvas element
    const canvas = document.getElementById('draw-canvas');
    // Set the width and the height to the better possible size
    let size = getCanvasSize();
    canvas.width = size;
    canvas.height = size;
    ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'white';
    ctx.fillStyle = 'white';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = getCtxSize();
}


function prepareCanvas()
{
    // Get the canvas element
    const canvas = document.getElementById('draw-canvas');
    // Resize the canvas element
    resizeCanvas();

    /** Mouse events for desktop computers. */
    canvas.addEventListener('mousedown', (e) => {
        if (!isModelLoaded)
            return ;

        drawing = true;
        stopPrediction = false;
        haveAlreadyPredicted = false;
        lastPosition = { x: e.offsetX, y: e.offsetY };
    });

    canvas.addEventListener('mouseout', async () => {
        let wasDrawing = drawing;
        drawing = false;
        await sleep(TIME_TO_WAIT_BEFORE_PREDICT_ON_MOUSE_OUT);
        if (stopPrediction)
            stopPrediction = false;
        else if (isModelLoaded && wasDrawing && !drawing)
            predict();
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!drawing)
            return ;
        ctx.beginPath();
        ctx.moveTo(lastPosition.x, lastPosition.y);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        lastPosition = { x: e.offsetX, y: e.offsetY };
    });

    canvas.addEventListener('mouseup', async () => {
        let wasDrawing = drawing;
        drawing = false;
        await sleep(TIME_TO_WAIT_BEFORE_PREDICT_ON_STOP_DRAWING);
        if (stopPrediction)
            stopPrediction = false;
        else if (isModelLoaded && wasDrawing && !drawing)
            predict();
    });


    /** Touch events for touch devices. */
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();

        if (!isModelLoaded)
            return ;

        drawing = true;
        haveAlreadyPredicted = false;
        stopPrediction = false;

        let clientRect = canvas.getBoundingClientRect();
        let touch = e.touches[0];
        let x = touch.pageX - clientRect.x;
        let y = touch.pageY - clientRect.y;
        lastPosition = { x, y };
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();

        if (!drawing)
            return ;

        let clientRect = canvas.getBoundingClientRect();
        let touch = e.touches[0];
        let x = touch.pageX - clientRect.x;
        let y = touch.pageY - clientRect.y;
        ctx.beginPath();
        ctx.moveTo(lastPosition.x, lastPosition.y);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastPosition = { x, y };
    });

    canvas.addEventListener('touchend', async () => {
        let wasDrawing = drawing;
        drawing = false;
        await sleep(TIME_TO_WAIT_BEFORE_PREDICT_ON_STOP_DRAWING);
        if (stopPrediction)
            stopPrediction = false;
        else if (isModelLoaded && wasDrawing && !drawing)
            predict();
    });
}


function createButton(innerText, selector, id, listener, disabled = false)
{
    const btn = document.createElement('BUTTON');
    btn.innerText = innerText;
    btn.id = id;
    btn.disabled = disabled;
    btn.addEventListener('click', listener);
    document.querySelector(selector).appendChild(btn);
}


function enableButton(selector)
{
    // Activates a button
    document.getElementById(selector).disabled = false;
}


function disableButton(selector)
{
    // Disables a button
    document.getElementById(selector).disabled = true;
}


async function loadModel()
{
    // Load the saved on MODEL_PATH and await the process to be complete
    model = await tf.loadLayersModel(MODEL_PATH);
    isModelLoaded = false;

    // Uncomment the line below if you want to see output on your browser console.
    // console.log("The model was loaded successfully!");

    printOutput();

    const canvas = document.getElementById('draw-canvas');
    canvas.title = 'Click and Hold to draw';
    canvas.style.cursor = 'crosshair';

    enableButton('clear-btn');
    isModelLoaded = true;
}


async function predict(showOutput = true)
{
    const canvas = document.getElementById('draw-canvas');

    // Aplicate the preprocessing transformations to be a valid input to the model
    const toPredict = tf.browser.fromPixels(canvas)
        .resizeNearestNeighbor([IMAGE_RESIZE, IMAGE_RESIZE]) // can use '.resizeBilinear' too
        .mean(2).pad([[IMAGE_PADDING_VALUE, IMAGE_PADDING_VALUE], [IMAGE_PADDING_VALUE, IMAGE_PADDING_VALUE]])
        .expandDims().expandDims(3).toFloat().div(255.0);
    
    // If the summed value of all pixels is equal to zero it means that nothing were drawn on the canvas
    const allPixelsSummedValue = toPredict.sum().dataSync()[0];

    if (!isModelLoaded || drawing || allPixelsSummedValue === 0)
    {
        if (allPixelsSummedValue === 0)
            printOutput('Click and Hold to draw');
        return ;
    }
    else
        disableButton('clear-btn');

    // HaveAlreadyPredicted prevents showing the same prediction to be predicted again
    if (haveAlreadyPredicted === false)
    {
        printOutput('Analyzing The Drawing(...)');
        await sleep(TIME_TO_WAIT_BEFORE_PREDICT_THE_IMAGE);
    } else
        haveAlreadyPredicted = false;


    if (stopPrediction === true)
    {
        stopPrediction = false;
        enableButton('clear-btn');
        printOutput();
    } else {
        // Prevents high usage of gpu
        tf.engine().startScope();

        // Predict the data and return an array with the probability of all possible outputs
        const prediction = model.predict(toPredict).dataSync();

        // Set the prediction to the output with the max probability (greater value) and shows it to the user
        const predictedValue = tf.argMax(prediction).dataSync();
        printOutput(`The number drawn is <strong>${predictedValue}</strong>` +
            ` (<strong>${NumberToWord[predictedValue]}</strong>)`);

        // Prevents high usage of gpu
        tf.engine().endScope();

        if (showOutput)
        {
            console.clear();
            // The greater probability represents the certainty of the model in the argmax prediction
            const greaterProbability = tf.max(prediction).dataSync()[0];
            console.log(` Prediction: ${predictedValue}\n Certainty ${(greaterProbability.toPrecision(4) * 100)}%`);
        }
    
        enableButton('clear-btn');
        haveAlreadyPredicted = true;
    }
}


(function init(message)
{
    // Prepares the canvas to be used
    prepareCanvas();
    resizePage();

    const clearBtn = document.getElementById('clear-btn');
    const output = document.getElementById('predict-output');
    const pipe = document.getElementById('pipeline');

    let width = `${getCanvasSize()}px`;
    output.style.width = width;
    pipe.style.width = width;

    clearBtn.addEventListener('click', () => {
        const size = getCanvasSize();

        ctx.clearRect(0, 0, size, size);

        if (isModelLoaded)
            printOutput();

        stopPrediction = true;
    });

    window.addEventListener('resize', () => {
        width = `${getCanvasSize()}px`;
        output.style.width = width;
        pipe.style.width = width;

        resizeCanvas();
        resizePage();

        if (isModelLoaded)
            printOutput();
    });

    // Load the model at last
    loadModel();

    console.log(message);
})('Welcome to the Digit Recognition Web App!');
