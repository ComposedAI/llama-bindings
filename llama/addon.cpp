#include <napi.h>
#include <llama.h>

class HelloAddon : public Napi::ObjectWrap<HelloAddon>
{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports)
    {
        Napi::Function func = DefineClass(env, "HelloAddon", {InstanceMethod("hello", &HelloAddon::Hello)});

        exports.Set("HelloAddon", func);
        return exports;
    }

    HelloAddon(const Napi::CallbackInfo &info) : Napi::ObjectWrap<HelloAddon>(info) {}

private:
    Napi::Value Hello(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        return Napi::String::New(env, "world");
    }
};

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    return HelloAddon::Init(env, exports);
}

NODE_API_MODULE(addon, Init)