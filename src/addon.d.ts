declare module "llama-bindings" {
  export type LLAMAModelParams = {
    gpuLayers?: number;
    vocabOnly?: boolean;
    useMmap?: boolean;
    useMlock?: boolean;
  };

  export function systemInfo(): string;

  export class LLAMAContext {
    constructor(modelPath: string, options?: LLAMAModelParams);
    encode(text: string): Uint32Array;
    batchEncode(input: string[]): Uint32Array;
    decode(tokens: Uint32Array): string;
    batchDecode(input: Uint32Array): string[];
    tokenBos(): number;
    tokenEos(): number;
    tokenN1(): number;
    evaluate(input: Uint32Array): Promise<number>;
  }
}
