import { min , max, CtxPosI } from "./common"


export interface SizeI {
    width: number;
    height: number;
}


// default export was not used on purpose
export class Canvas {
    protected readonly canvas: HTMLCanvasElement;
    public readonly ctx: CanvasRenderingContext2D;
    private lastCtxPos: CtxPosI;
    public drawing: boolean;

    constructor(
        protected readonly selector: string,
        protected readonly canvasSize: SizeI,
        protected readonly ctxSize: number
    ) {
        this.lastCtxPos = {x: 0, y: 0};
        this.drawing = false;
        this.canvas = document.getElementById(selector) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d');
    }

    getCanvasElement = (): HTMLCanvasElement => {
        return document.getElementById(this.selector) as HTMLCanvasElement;
    }

    getCtxElement = (): CanvasRenderingContext2D => {
        return this.getCanvasElement().getContext('2d');
    }

    setLastCtxPosition = (pos: CtxPosI) => {
        this.lastCtxPos = pos;
    }

    getLastCtxPosition = () => {
        return this.lastCtxPos;
    }

    canvasBetterSize = (paddingIncrement: number = 30): number => {
        const {width, height} = this.canvasSize;
        const maxSize = max(width, height);
        const {innerWidth: innerW, outerWidth: outerW, ...o} = window;
        const betterWidth: number = min(innerW, outerW) || innerW;
        return betterWidth > (maxSize + paddingIncrement) ? 
            maxSize : (betterWidth - paddingIncrement);
    }

    ctxBetterSize = (): number => {
        const {width: canvasW, height: canvasH} = this.canvasSize;
        const maxCanvasSize = max(canvasW, canvasH);
        return (this.canvasBetterSize()*this.ctxSize) / maxCanvasSize;
    }

    private setUpCtx = (
        strokeStyle: string = 'white',
        fillStyle: string = 'white',
        lineJoin: CanvasLineJoin = 'round',
        lineCap: CanvasLineCap = 'round'
    ) => {
        this.ctx.strokeStyle = strokeStyle;
        this.ctx.fillStyle = fillStyle;
        this.ctx.lineJoin = lineJoin;
        this.ctx.lineCap = lineCap;
    }
    
    resize = () => {
        const canvasSize = this.canvasBetterSize();
        const ctxSize = this.ctxBetterSize();
        this.canvas.width = canvasSize;
        this.canvas.height = canvasSize;
        this.ctx.lineWidth = ctxSize;
        this.setUpCtx();
    }

    setEvent = (type: string, listener: any) => {
        this.canvas.addEventListener(type, listener);
    }

    clear = () => {
        const canvasSize = this.canvas.width;
        this.ctx.clearRect(0, 0, canvasSize, canvasSize);
    }
}