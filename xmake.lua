add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
end

-- MetaX (MACA) --
option("mx-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU")
option_end()

if has_config("mx-gpu") then
    add_defines("ENABLE_METAX_API")
end

includes("xmake/metax.lua")

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    if has_config("nv-gpu") then
        add_rules("cuda")
        add_cugencodes("native")
        add_files("src/device/nvidia/*.cu", "src/ops/*/nvidia/*.cu")
        add_cuflags("--compiler-options", "-fPIC", {force = true})
        add_links("cudart", "cublas")
        add_linkdirs("/usr/local/cuda/lib64")
    end
    if has_config("mx-gpu") then
        add_includedirs("/opt/maca/include")
        add_linkdirs("/opt/maca/lib")
        add_links("mcblas")
        before_build(function (target)
            local mxcc = "/opt/maca/mxgpu_llvm/bin/mxcc"
            local cu_files = {}
            for _, f in ipairs(os.files("src/device/metax/*.cu")) do table.insert(cu_files, f) end
            for _, f in ipairs(os.files("src/ops/*/metax/*.cu")) do table.insert(cu_files, f) end
            for _, sourcefile in ipairs(cu_files) do
                local objectfile = target:objectfile(sourcefile)
                os.mkdir(path.directory(objectfile))
                print("compiling.maca %s", sourcefile)
                os.vrunv(mxcc, {"-c", sourcefile, "-o", objectfile, "-fPIC", "-std=c++17",
                    "-Iinclude", "-I/opt/maca/include", "-DENABLE_METAX_API"})
                table.insert(target:objectfiles(), objectfile)
            end
        end)
    end
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()