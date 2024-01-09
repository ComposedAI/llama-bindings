#include <napi.h>
#include <llama.h>

Napi::Value systemInfo(const Napi::CallbackInfo &info)
{
    return Napi::String::From(info.Env(), llama_print_system_info());
}

void FinalizeModel(Napi::Env env, llama_model *data)
{
    // Clean up the llama_model object
    llama_free_model(data);
}

Napi::Value LlamaLoadModelFromFile(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 2)
    {
        Napi::TypeError::New(env, "Wrong number of arguments")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::String pathModel = info[0].As<Napi::String>();
    Napi::Object params = info[1].As<Napi::Object>();

    struct llama_model_params llamaParams;
    // Fill llamaParams from params object

    struct llama_model *result = llama_load_model_from_file(pathModel.Utf8Value().c_str(), llamaParams);

    // Wrap the pointer in a Napi::External object and return it
    return Napi::External<llama_model>::New(env, result);
}

Napi::Value LlamaModelDesc(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1)
    {
        Napi::TypeError::New(env, "Wrong number of arguments")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::External<llama_model> model = info[0].As<Napi::External<llama_model>>();
    char buf[1024]; // Adjust size as needed
    llama_model_desc(model.Data(), buf, sizeof(buf));

    return Napi::String::New(env, buf);
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set(Napi::String::New(env, "getSystemInfo"),
                Napi::Function::New(env, systemInfo));
    exports.Set("loadModelFromFile", Napi::Function::New(env, LlamaLoadModelFromFile));
    exports.Set("getModelDesc", Napi::Function::New(env, LlamaModelDesc));
    return exports;
}

NODE_API_MODULE(addon, Init)
