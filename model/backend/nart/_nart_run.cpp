//
// _nart_run - nart backend runner pybind11 module
//

#include <dlfcn.h>
#include <sys/time.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "art/parade.h"


namespace py = pybind11;

/// START defines from NART/python/src/art/module.cpp
using std::string;
using std::unordered_map;
using std::vector;

/**
 * @brief A context that manages workspaces requried by a parade.
 */
class RuntimeContext {
public:
    RuntimeContext(const vector<string> &modules, const string &input_module);
    ~RuntimeContext();

    /**
     * @brief Returns the names of workspaces created from loaded modules.
     *  Users can use this method to check which workspaces are created
     * successfully.
     */
    vector<string> loaded_workspaces() const;
    workspace_t *const *workspace_data() const;
    const mem_tp *get_input_mem_tp() const;

private:
    vector<string> modules;
    vector<workspace_t *> workspaces;
    const mem_tp *input_mem_tp;

    // names of modules which are failed to load.
    static std::unordered_set<string> load_failed;
};

/**
 * @brief A wrapper class of art parade. This class servers to ease the usage of
 * parade in cpp.
 */
class Parade {
public:
    Parade(struct buffer_t *buf, const RuntimeContext &ctx);
    ~Parade();

    /**
     * @brief Reshape the current Parade.
     */
    bool reshape(const unordered_map<string, vector<uint32_t>> &shapes);

    /**
     * @brief Do parade preparation after reshape is done, initialization any
     * resources required by operators, and allocate memory spaces for tensors.
     */
    void prepare();

    /**
     * @brief Do inference .
     */
    void forward();

    /**
     * @brief Get a tensor_t by it's name.
     */
    tensor_t *get_tensor_by_name(const std::string &name);

    /**
     * @brief Get outputs .
     */
    bool get_output(size_t *output_count, tensor_array_t *outputs);

    const vector<string> &get_input_names() const { return this->input_names; }
    const vector<string> &get_output_names() const { return this->output_names; }

private:
    parade_t *raw_parade;
    unordered_map<string, tensor_t *> tensor_by_name;
    vector<string> input_names;
    vector<string> output_names;
};

void Parade::prepare() { parade_prepare(this->raw_parade); }

void Parade::forward() { parade_run(this->raw_parade); }

tensor_t *Parade::get_tensor_by_name(const std::string &name)
{
    if (tensor_by_name.find(name) != tensor_by_name.end())
        return tensor_by_name[name];
    else
        return nullptr;
}

bool Parade::get_output(size_t *output_count, tensor_array_t *outputs)
{
    return parade_get_output_tensors(this->raw_parade, output_count, outputs);
}

// placeholder
// Parade::~Parade(){}
// Parade::Parade(struct buffer_t *buf, const RuntimeContext &ctx){}

unordered_map<int, string> DTYPE_NAME = {
    {dtINT8,     "int8"   },
    { dtINT16,   "int16"  },
    { dtINT32,   "int32"  },
    { dtINT64,   "int64"  },
    { dtUINT8,   "uint8"  },
    { dtUINT16,  "uint16" },
    { dtUINT32,  "uint32" },
    { dtUINT64,  "uint64" },
    { dtFLOAT16, "float16"},
    { dtFLOAT32, "float32"},
    { dtFLOAT64, "float64"},
    { dtBOOL,    "bool"   },
};

/// END defines from NART/python/src/art/module.cpp


// modified from Parade's .run() method def in NART/python/src/art/module.cpp
py::list perf_parade_run(py::handle obj, py::dict &inputs, py::dict &outputs, size_t run_times = 10, size_t warm_up_times = 3) {
    // this method accepts a dict of numpy.array, the do inference
    // with their content as tensor data. steps:
    // 1. check all inputs/outputs are given and their shape/dtype matches
    // the parade's input tensor.
    // 2. copy data to parade's input tensor's mem.
    // 3. do inference
    // 4. copy output tensors' data to ndarray in outputs.

    // TODO: maybe unsafe
    /* not work: RuntimeError: Unable to cast Python instance of
         type <class 'nart.art._art.Parade'> to C++ type 'Parade' */
    // Parade &parade = obj.cast<Parade &>();

    /* I don't care, force cast */
    auto inst = reinterpret_cast<py::detail::instance *>(obj.ptr());
    Parade &parade = *reinterpret_cast<Parade*>(inst->simple_value_holder[0]);

    using std::string;
    using std::vector;

    const vector<string> &input_tensor_names = parade.get_input_names();
    for (const string &name : input_tensor_names) {
        assert(inputs.contains(name) && "some input not given when calling parade.run");
        tensor_t *tensor = parade.get_tensor_by_name(name);
        py::array input = inputs[name.c_str()].cast<py::array>();
        if (tensor->shape.dim_size != input.ndim()) {
            throw py::value_error("input dim size not match");
        }
        auto np_shape = input.shape();
        for (pybind11::ssize_t idx = 0; idx < input.ndim(); idx++) {
            if (tensor->shape.dim[idx] != np_shape[idx]) {
                throw py::value_error("input shape not match");
            }
        }
        if (DTYPE_NAME.at(tensor->dtype) != py::str(input.dtype()).cast<string>()) {
            throw py::value_error("input dtype not match");
        }
    }

    const vector<string> &output_tensor_names = parade.get_output_names();
    for (const string &name : output_tensor_names) {
        assert(
            outputs.contains(name) && "some output not given when calling parade.run");
        tensor_t *tensor = parade.get_tensor_by_name(name);
        py::array output = outputs[name.c_str()].cast<py::array>();
        if (tensor->shape.dim_size != output.ndim()) {
            throw py::value_error("output dim size not match");
        }
        auto np_shape = output.shape();
        for (pybind11::ssize_t idx = 0; idx < output.ndim(); idx++) {
            if (tensor->shape.dim[idx] != np_shape[idx]) {
                throw py::value_error("output shape not match");
            }
        }
        if (DTYPE_NAME.at(tensor->dtype) != py::str(output.dtype()).cast<string>()) {
            throw py::value_error("output dtype not match");
        }
    }

    for (const string &name : input_tensor_names) {
        py::array input = inputs[name.c_str()];
        input = py::cast<py::array>(
            py::module::import("numpy").attr("ascontiguousarray")(input));
        tensor_t *tensor = parade.get_tensor_by_name(name);
        memcpy(
            mem_cpu_data(tensor->mem), input.data(),
            datatype_sizeof(tensor->dtype) * shape_count(&tensor->shape));
    }

    for (size_t i = 0; i < warm_up_times; i++) {
        parade.forward(); // warm up
    }

    struct timeval start, end;
    vector<double> times(run_times);

    for (size_t i = 0; i < run_times; i++){
        gettimeofday(&start, NULL);

        parade.forward();

        gettimeofday(&end, NULL);
        times[i] = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.;
    }


    tensor_array_t outputs_;
    size_t output_count;
    parade.get_output(&output_count, &outputs_);
    for (size_t i = 0; i < output_count; ++i) {
        memcpy(
            outputs.cast<py::dict>()[outputs_[i]->name]
                .cast<py::array>()
                .mutable_data(),
            mem_cpu_data(outputs_[i]->mem),
            datatype_sizeof(outputs_[i]->dtype) * shape_count(&outputs_[i]->shape));
    }

    return py::cast(times);
}


PYBIND11_MODULE(_nart_run, m) {
    m.doc() = "nart backend runner module";
    m.def("perf_parade_run", &perf_parade_run, "perf parade_run(), returns a list of run times in ms",
        py::arg("Parade"), py::arg("inputs"), py::arg("outputs"), py::arg("run_times") = 10, py::arg("warm_up_times") = 3);
}
