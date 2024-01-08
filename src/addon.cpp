#include <napi.h>
#include <llama.h>

class LLAMAModel : public Napi::ObjectWrap<LLAMAModel>
{
public:
    llama_model_params model_params;
    llama_model *model;

    LLAMAModel(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LLAMAModel>(info)
    {
        model_params = llama_model_default_params();

        // Get the model path
        std::string modelPath = info[0].As<Napi::String>().Utf8Value();

        if (info.Length() > 1 && info[1].IsObject())
        {
            Napi::Object options = info[1].As<Napi::Object>();

            if (options.Has("gpuLayers"))
            {
                model_params.n_gpu_layers = options.Get("gpuLayers").As<Napi::Number>().Int32Value();
            }

            if (options.Has("vocabOnly"))
            {
                model_params.vocab_only = options.Get("vocabOnly").As<Napi::Boolean>().Value();
            }

            if (options.Has("useMmap"))
            {
                model_params.use_mmap = options.Get("useMmap").As<Napi::Boolean>().Value();
            }

            if (options.Has("useMlock"))
            {
                model_params.use_mlock = options.Get("useMlock").As<Napi::Boolean>().Value();
            }
        }

        llama_backend_init(false);
        model = llama_load_model_from_file(modelPath.c_str(), model_params);

        if (model == NULL)
        {
            Napi::Error::New(info.Env(), "Failed to load model").ThrowAsJavaScriptException();
            return;
        }
    }

    ~LLAMAModel()
    {
        llama_free_model(model);
    }

    static void init(Napi::Object exports)
    {
        exports.Set("LLAMAModel", DefineClass(exports.Env(), "LLAMAModel", {}));
    }
};

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    LLAMAModel::init(exports);
    return exports;
}

NODE_API_MODULE(addon, Init)