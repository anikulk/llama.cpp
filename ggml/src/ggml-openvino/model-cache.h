#pragma once

// Compiled blobs include weights. Cache-only execution skips weight uploads and graph compilation.
// On NPU, GGML_OPENVINO_SELF_CONTAINED_BLOB=1 exports embedded-weight prefill/decode
// blobs. A warm hit skips weight requantization, graph conversion, and compilation,
// at the cost of roughly model-size storage per blob and separate NPUW weight banks.
// After both blobs exist, GGML_OPENVINO_SELF_CONTAINED_MMAP=1 additionally wraps the
// GGUF mmap instead of allocating the private OpenVINO model buffer. This mode is
// import-only and fails clearly if either required blob is missing or invalid.
// GGML_OPENVINO_NPU_WEIGHTLESS_BANK=1 is the same raw-weight flow, but exports weightless
// prefill/decode blobs plus ONE weights file per model, shared by all its blob pairs. Warm import
// reads the weights from that file (ov::weights_path) and NPUW deduplicates the two models into one
// bank, so the weights are stored once on disk and resident once in the bank.

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ov {
class Model;
class Node;
}

bool ggml_openvino_model_cache_only();
void ggml_openvino_model_cache_init();

struct ggml_openvino_source_mapping {
    uintptr_t begin;
    uintptr_t end;
    uint64_t offset;
    uint64_t identity;
};

// Identify mmap weights without reading their pages. Cache mappings for one buffer lifetime.
uint64_t ggml_openvino_source_fingerprint(const void * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings);
uint64_t ggml_backend_openvino_weight_fingerprint(const ggml_tensor * tensor);

// Returns the compiled-model cache directory, or empty if unset.
std::string ggml_openvino_model_cache_dir();
std::string ggml_openvino_model_cache_temp_path(const std::string & path);

// Hash graph structure, source weight identities, configuration, and OpenVINO version.
uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg,
                                         const std::string & graph_signature);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint);

// Record weight metadata and source identities. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              const std::vector<std::string> & inputs,
                                              const std::vector<std::string> & outputs);

// Require all weight metadata and source identities to match the manifest.
bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs);

// GGML_OPENVINO_NPU_WEIGHTLESS_BANK: one weights file per model and weight settings, shared by every
// prefill/decode blob pair of that model; each pair records the file it uses in a .weights.ref.

// Identity of the extracted weights: source weight identities plus the settings that change their bytes.
uint64_t ggml_openvino_model_cache_weights_fingerprint(const ggml_cgraph * cgraph, uint64_t weight_cfg);

// Path of the reference file naming the weights file a prefill/decode blob pair was exported with.
std::string ggml_openvino_model_cache_weights_ref_path(const std::string & dir, uint64_t prefill_fp, uint64_t decode_fp);

// Lay the weight Constants (reachable from |weight_roots|) of |models| out in one file, retag their
// WeightlessCacheAttribute with the file offsets, untag all other Constants so they stay embedded in the blob,
// and write the file unless an identical one exists. Call before compiling. Returns the file path, empty on error.
std::string ggml_openvino_model_cache_prepare_weights(const std::string & dir,
                                                      uint64_t weights_fp,
                                                      const std::vector<std::shared_ptr<ov::Node>> & weight_roots,
                                                      const std::vector<std::shared_ptr<ov::Model>> & models);

// True if the weights file is complete (its size matches the size record written after it).
bool ggml_openvino_model_cache_weights_valid(const std::string & path);

bool ggml_openvino_model_cache_write_weights_ref(const std::string & ref_path, const std::string & weights_path);

// Weights file named by a reference file, or empty if the reference is missing or malformed.
std::string ggml_openvino_model_cache_read_weights_ref(const std::string & dir, const std::string & ref_path);
