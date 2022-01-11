import { Button } from './common';
import { OutputLabel } from './common';
import { Logger, sleep } from './common';
import { Canvas } from './canvas';


export interface MPI {
    value: number;
    name: string;
    certainty: number;
    userDrawing?: any; // tf tensor output type
}


const DigitNames = {
    0: 'Zero', 1: 'One',
    2: 'Two', 3: 'Three',
    4: 'Four', 5: 'Five',
    6: 'Six',7: 'Seven',
    8: 'Eight', 9: 'Nine',
};


export class Model {
    private _model: any;
    private predictions: MPI[];
    private readonly inputShape: number[];
    private readonly paddingShape: number[][];
    private modelWasLoaded: boolean;
    public lastDrawPredicted: boolean;
    private halt: boolean;
    private postHaltProcedure?: Function;

    constructor(
        private readonly path: string,
        private readonly canvas: Canvas,
        private readonly eraseButton: Button,
        private readonly outputLabel: OutputLabel,
        private readonly logger: Logger
    ) {
        this.predictions = [];
        this.modelWasLoaded = false;
        this.halt = false;
        this.lastDrawPredicted = true;
        const padding = 2;
        const inputSize = 36; 
        const shapeSize = inputSize - 2 * padding;
        this.inputShape = [shapeSize, shapeSize];
        this.paddingShape = [
            [padding, padding],
            [padding, padding]
        ];
    }

    isLoaded = (): boolean => this.modelWasLoaded;
    
    load = async () => {
        this.eraseButton.disable();
        this._model = await tf.loadLayersModel(this.path);
        this.modelWasLoaded = this._model !== undefined;
        this.logger.writeLog('Model.load: ' + (this.modelWasLoaded ?
            "The model was loaded successfully!" :
            "Error: The model was not loaded, try to reload the page.")
        );
        if (this.modelWasLoaded === true) {
            // Predict the empty canvas at least one time,
            // because the first prediction is the slowest one.
            this.makePrediction(this.getInputTensor());
            this.canvas.getCanvasElement().style.cursor = 'crosshair';
            this.eraseButton.enable();
            this.outputLabel.defaultMessage();
        }
    }

    private getInputTensor = (): any => {
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

    analyzeDrawing = async (wait: number = 150, returnDrawing: boolean = false, save: boolean = false): Promise<MPI> => {
        this.outputLabel.write("<<< Analyzing your Drawings >>>");
        this.eraseButton.disable();

        const inputTensor = this.getInputTensor();

        if (this.modelWasLoaded === false || this.canvas.drawing === true) {
            this.activateHalt(() => {
                this.eraseButton.enable();
                this.outputLabel.defaultMessage();
                this.logger.writeLog('Model.analyzeDrawing: ' + (this.modelWasLoaded ?
                    'model was not loaded yet, prediction canceled!' : 
                    'user is drawing, prediction canceled!')
                );
            });
        } else if (inputTensor.sum().dataSync()[0] === 0) {
            this.activateHalt(() => {
                this.eraseButton.enable();
                this.outputLabel.write("<div id='output-text'><strong>TIP</strong>:"+
                    "  Click and Hold to draw.<\div>"
                );
                this.logger.writeLog('Model.analyzeDrawing: canvas has no drawings, prediction canceled!');
            });
        }

        if (this.checkHalt()) {
            return ;
        } else {
            await sleep(this.checkLastDrawPredicted() === false ? wait : 0);
            this.lastDrawPredicted = true;
            const prediction = this.makePrediction(inputTensor, returnDrawing);
            if (save === true) { this.predictions.push(prediction); }
            this.outputLabel.write("Analysis finished.");
            this.eraseButton.enable();
            return prediction;
        }
    }

    private makePrediction = <T>(inputTensor: T, returnDrawing: boolean = false): MPI => {
        // This prevents high usage of GPU
        tf.engine().startScope();
        const outputTensor = this._model.predict(inputTensor).dataSync();
        const predictedValue = tf.argMax(outputTensor).dataSync();
        const predictionValueName = DigitNames[predictedValue];
        const predictionCertainty = tf.max(outputTensor).dataSync();
        tf.engine().endScope();

        const prediction: MPI = {
            name: predictionValueName,
            value: predictedValue,
            certainty: predictionCertainty,
            userDrawing: returnDrawing ?
                        inputTensor :undefined,
        }

        return prediction;
    }

    activateHalt = (postHaltProcedure?: Function): void => {
        this.halt = true;
        if (postHaltProcedure !== undefined) {
            this.postHaltProcedure = postHaltProcedure;
        }
    }

    deactivateHalt = () => {
        this.halt = false;
        this.postHaltProcedure = undefined;
    }

    checkHalt = (): boolean => {
        const halt = this.halt;

        if (halt === true) {
            if (this.postHaltProcedure !== undefined) {
                this.postHaltProcedure();
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