let model;
let isModelLoaded = false;
const IMAGE_SIZE = 28;
const MODEL_PATH = './tfjs/DigitRec/model.json';
const INITIAL_MESSAGE = 'Draw any digit between <strong>0</strong> to <strong>9</strong>';

let lastPosition = {x: 0, y: 0};
let drawing = false;

const MAX_CANVAS_SIZE = 400;
const MAX_CTX_SIZE = 22;
const RESIZE_SUB_FACTOR = 30;
let ctx;


function min(a, b, ...args)
{
    let minValue;
    if (args.length > 0) {
        args.push(a, b);
        args.sort();
        minValue = args[0];
    } else {
        minValue = a < b ? a : b;
    }
    return minValue;
}

function getCanvasSize()
{
    let win_size = min(window.innerWidth, window.outerWidth);
    let size = win_size > (MAX_CANVAS_SIZE + RESIZE_SUB_FACTOR) ?
        MAX_CANVAS_SIZE : (win_size - RESIZE_SUB_FACTOR);
    return size;
}


function getCtxSize()
{
    let win_size = min(window.innerWidth, window.outerWidth);
    let size = win_size > 280 ? MAX_CTX_SIZE : win_size/15;
    return size;
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
    canvas.width = canvas.height = getCanvasSize();
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
        drawing = true;
        lastPosition = { x: e.offsetX, y: e.offsetY };
    });
    canvas.addEventListener('mouseout', async () => {
        let wasDrawing = drawing;
        drawing = false;
        await sleep(1100);
        if (wasDrawing && !drawing && isModelLoaded)
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
        drawing = false;
        await sleep(650);
        if (!drawing && isModelLoaded)
            predict();
    });

    /** Touch events for touch devices. */
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        drawing = true;
        let clientRect = canvas.getBoundingClientRect();
        let touch = e.touches[0];
        let x = touch.pageX - clientRect.x;
        let y = touch.pageY - clientRect.y;
        lastPosition = { x, y };
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();

        // If not drawing stop the execution of this funtion here.
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
        drawing = false;
        // Sleeps for some miliseconds and if not drawing predict the image in canvas
        await sleep(600);
        if (!drawing && isModelLoaded)
            predict();
    });
}


function createButton(innerText, selector, id, listener, disabled = false) {
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
    isModelLoaded = true;

    // Uncomment the line below if you want to see output on your browser console.
    // console.log("The model was loaded successfully!");

    const p = document.getElementById("predict-output");
    p.innerHTML = INITIAL_MESSAGE;
}


async function predict()
{
    // If the model isn't loaded stops the execution of this function here
    if (!isModelLoaded)
        return ;

    const p = document.getElementById('predict-output');
    p.innerText = 'Predicting...';
    await sleep(350);

    tf.engine().startScope();
    // Get the canvas element with the drawing
    const canvas = document.getElementById('draw-canvas');
    // Aplicate the preprocessing transformations for being able to be a valid input to the model
    const toPredict = tf.browser.fromPixels(canvas)
        .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
        .mean(2).expandDims().expandDims(3).toFloat().div(255.0);
    // Predict the data and return an array with the probability of all possible outputs
    const prediction = model.predict(toPredict).dataSync();
    // Set the prediction to the output with the max probability (greater value) and shows it to the user
    p.innerHTML = `The Predicted value is: <strong>${tf.argMax(prediction).dataSync()}</strong>`;
    tf.engine().endScope();
}


/** Automatic call to function init */
(function init() {
    // Prepares the canvas to be used
    prepareCanvas();

    const p = document.getElementById('predict-output');
    const pipe = document.getElementById('pipeline');
    p.style.width = pipe.style.width = `${getCanvasSize()}px`;

    // Create the clear button along with its event
    createButton('Clear', '#pipeline', 'clear-btn', () => {
        let size = getCanvasSize();
        ctx.clearRect(0, 0, size, size);
        if (isModelLoaded)
            p.innerHTML = INITIAL_MESSAGE;
    });

    /** When resizing the window some other elements must be resized also */
    window.addEventListener('resize', () => {
        p.style.width = pipe.style.width = `${getCanvasSize()}px`;
        resizeCanvas();
        if (isModelLoaded)
            p.innerHTML = INITIAL_MESSAGE;
    });

    // Load the model at last
    loadModel();
})();
