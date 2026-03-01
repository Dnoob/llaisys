-- MetaX MACA custom build rule
-- mxcc compiles .cu files similar to nvcc
rule("maca")
    set_extensions(".cu")
    on_build_file(function (target, sourcefile, opt)
        import("utils.progress")
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))

        progress.show(opt.progress, "${color.build.object}compiling.maca %s", sourcefile)

        local mxcc = "/opt/maca/mxgpu_llvm/bin/mxcc"
        local flags = {"-c", sourcefile, "-o", objectfile, "-fPIC"}

        for _, dir in ipairs(target:get("includedirs")) do
            table.insert(flags, "-I" .. dir)
        end
        table.insert(flags, "-Iinclude")

        for _, define in ipairs(target:get("defines")) do
            table.insert(flags, "-D" .. define)
        end

        os.vrunv(mxcc, flags)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()
