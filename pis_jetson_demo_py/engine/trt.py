from typing import Union

import numpy as np
from loguru import logger

logger.info("Loading TRT libraries")
import pycuda.autoinit  # pylint: disable=W0611
import pycuda.driver as cuda
import tensorrt as trt
from tensorrt.tensorrt import Dims  # pylint: disable=E0611

logger.info("Successfully loaded TRT pycuda libraries")


class TRTInfer:
    def __init__(self, engine_path: str, dynamic_max_batch: Union[int, None] = None):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.dynamic_batch = False
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)

            if shape[0] == -1:
                self.dynamic_batch = True
                assert not (
                    dynamic_max_batch is None
                ), "Dynamic batch model detected (batch size -1)."
                shape = (dynamic_max_batch, *shape[1:])

                if self.engine.binding_is_input(i):
                    # only consider input node
                    self.context.set_binding_shape(i, Dims(shape))
                logger.debug(
                    f"Engine bindings (Rebound) - name: {name} dtype: {dtype} shape: {shape}"
                )
            else:
                assert (
                    dynamic_max_batch is None
                ), "Static batch model detected but dynamic batch specified."
                logger.debug(
                    f"Engine bindings - name: {name} dtype: {dtype} shape: {shape}"
                )

            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            logger.debug(f"Engine bindings: {binding}")
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs

    def multi_stage_infer(self, batch):
        # Split batch over supported batch size
        infer_iterations = batch.shape[0] // self.batch_size
        if batch.shape[0] % self.batch_size != 0:
            infer_iterations += 1

        outputs = []
        for _, _ in self.output_spec():
            outputs.append([])

        for infer_idx in range(infer_iterations):
            b_start = self.batch_size * infer_idx
            b_end = min(batch.shape[0], self.batch_size * (infer_idx + 1))
            b_items = b_end - b_start

            current_batch = batch[b_start:b_end]
            current_outputs = self.infer(current_batch)
            for idx, output_node in enumerate(current_outputs):
                outputs[idx].append(output_node[:b_items])

        outputs = [np.concatenate(node_outputs, axis=0) for node_outputs in outputs]
        return outputs

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        """

        if batch.shape[0] > self.batch_size:
            return self.multi_stage_infer(batch)

        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        for idx, item in enumerate(outputs):
            cuda.memcpy_dtoh(item, self.outputs[idx]["allocation"])

        # Filter out batch results
        if self.dynamic_batch:
            for idx, output in enumerate(outputs):
                outputs[idx] = output[: batch.shape[0], :]

        # Return the results.
        return outputs
