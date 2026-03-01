target("llaisys-device-nvidia")
    set_kind("object")
    add_rules("cuda")
    add_cugencodes("native")
    add_includedirs("../include")
    add_cuflags("--compiler-options", "-fPIC", {force = true})

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("object")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    add_cugencodes("native")
    add_includedirs("../include")
    add_cuflags("--compiler-options", "-fPIC", {force = true})
    add_links("cublas")

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
