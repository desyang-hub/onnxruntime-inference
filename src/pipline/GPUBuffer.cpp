#include "pipline/GPUBuffer.h"

#include "pipline/TaskContext.h"

GPUBuffer::GPUBuffer(const TaskContext& context) :
    g_input(context.num_input_elements),
    g_output(context.num_output_elements) {

    input_tensor = Ort::Value::CreateTensor<float>(
        context.active_mem_info,
        g_input.get(),
        context.num_input_elements,
        context.input_shape.data(),
        context.input_shape.size()
    );

    output_tensor = Ort::Value::CreateTensor<float>(
        context.active_mem_info,
        g_output.get(),
        context.num_output_elements,
        context.output_shape.data(),
        context.output_shape.size()
    );
}