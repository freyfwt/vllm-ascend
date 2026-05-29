add_op_to_compiled_list()
set(gumbel_sample_from_candidates_depends sample/sampling_common PARENT_SCOPE)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        gumbel_sample_from_candidates_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME GumbelSampleFromCandidates
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE gumbel_sample_from_candidates ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../sampling_common
    )
endif()
