declare module "llama-bindings" {
  export function getSystemInfo(): string;
  export function loadModelFromFile(pathModel: string, params: object): any;
  export function getModelDesc(model: any): string;
}
