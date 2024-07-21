import { Button } from './common';
import { OutputSection } from './common';
import { Logger, sleep } from './common';
import { Canvas } from './canvas';

import  {
    IPrediction,
    IModelSettings,
    ModelPaddingType
} from './types';


const INPUT_SIZE = 36;


const DigitNames = {
    0: 'Zero', 1: 'One',
    2: 'Two', 3: 'Three',
    4: 'Four', 5: 'Five',
    6: 'Six',7: 'Seven',
    8: 'Eight', 9: 'Nine',
};


export class Model {
    private mnet: any;
    private readonly log: Logger;
    private predictions: IPrediction[];
    private readonly inputShape: number[];
    private readonly paddingShape: number[][];
    private readonly path: string;
    private modelWasLoaded: boolean;

    private __postHaltProcedure: Function;
    private __halt: boolean;

    public lastDrawPredicted: boolean = true;

    constructor(
        private readonly settings: IModelSettings,
        private readonly canvas: Canvas,
        private readonly eraseButton: Button,
        private readonly outputLabel: OutputSection,
    ) {
        const { padding, path } = this.settings;

        const size = INPUT_SIZE - 2 * padding;

        this.inputShape = [size, size];

        this.paddingShape = [
            [padding, padding],
            [padding, padding]
        ];

        this.path = path;
        this.predictions = [];
        this.log = Logger.getInstance();
    }

    isLoaded = (): boolean => this.modelWasLoaded;

    load = async () => {
        this.eraseButton.disable();
        this.mnet = await tf.loadLayersModel(this.path);
        this.modelWasLoaded = this.mnet !== undefined;
        if (this.modelWasLoaded) {
            // Predict the empty canvas at least one time,
            // because the first prediction is the slowest one.
            this.predict(this.getInputTensor());
            this.canvas.getCanvasElement().style.cursor = 'crosshair';
            this.eraseButton.enable();
            this.log.writeLog("The model was loaded successfully!");
            this.outputLabel.defaultMessage();
        } else {
            this.log.writeLog("Error: The model was not loaded, try to reload the page.");
            this.outputLabel.write("Digits classification model couldn't be loaded!");
        }
    }

    private getInputTensor = () => {
        return tf.browser
            .fromPixels(this.canvas.getCanvasElement())
            .resizeBilinear(this.inputShape)
            .mean(2)
            .pad(this.paddingShape)
            .expandDims()
            .expandDims(3)
            .toFloat()
            .div(255.0);
    }

    analyzeDrawing = async (save: boolean = false): Promise<IPrediction> => {
        this.eraseButton.disable();
        this.outputLabel.write("Analyzing.");

        const inputTensor = this.getInputTensor();

        if (this.modelWasLoaded === false || this.canvas.drawing === true) {
            this.activateHalt(() => {
                this.eraseButton.enable();
                this.outputLabel.defaultMessage();
                this.log.writeLog('Model.analyzeDrawing: ' + (this.modelWasLoaded ?
                    'model was not loaded yet, prediction canceled!' : 
                    'user is drawing, prediction canceled!')
                );
            });
        } else if (inputTensor.sum().dataSync()[0] === 0) {
            this.activateHalt(() => {
                this.eraseButton.enable();
                this.outputLabel.write(`<div id='output-text'><strong>TIP</strong>: Click and Hold to draw.<\div>`);
                this.log.writeLog('Model.analyzeDrawing: canvas has no drawings, prediction canceled!');
            });
        }

        if (!this.checkHalt()) {
            const sleepInterval = this.settings.sleepMilisecsOnPrediction;

            if (!this.checkLastDrawPredicted()) {
                this.outputLabel.write("Analyzing..");
                await sleep(sleepInterval);
            }

            this.lastDrawPredicted = true;
            const prediction = this.predict(inputTensor);

            this.outputLabel.write("Analyzing...");

            if (save === true) { this.predictions.push(prediction); }
            this.outputLabel.write("Got the results!");
            this.eraseButton.enable();
            return prediction;
        }
    }

    private predict = <T>(inputTensor: T): IPrediction => {
        // This prevents high usage of GPU
        tf.engine().startScope();
        const outputTensor = this.mnet.predict(inputTensor).dataSync();
        const predictedValue = tf.argMax(outputTensor).dataSync();
        const predictionValueName = DigitNames[predictedValue];
        const predictionCertainty = tf.max(outputTensor).dataSync();
        tf.engine().endScope();

        return {
            name: predictionValueName,
            value: predictedValue,
            certainty: predictionCertainty,
        }
    }

    activateHalt = (postHaltProcedure?: Function): void => {
        this.__halt = true;
        if (postHaltProcedure !== undefined) {
            this.__postHaltProcedure = postHaltProcedure;
        }
    }

    deactivateHalt = () => {
        this.__halt = false;
        this.__postHaltProcedure = undefined;
    }

    checkHalt = (): boolean => {
        const halt = this.__halt;

        if (halt === true) {
            if (this.__postHaltProcedure !== undefined) {
                this.__postHaltProcedure();
            }

            this.deactivateHalt();
        }

        return halt;
    }

    checkLastDrawPredicted = (): boolean => {
        const lastDrawPredicted = this.lastDrawPredicted;

        if (lastDrawPredicted === true) {
            this.lastDrawPredicted = false;
        }

        return lastDrawPredicted;
    }
};