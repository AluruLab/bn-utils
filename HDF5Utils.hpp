/**
 * @file HDF5Utils.hpp
 * @brief Declaration of functions for common set HDF5 operations.
 * @author Tony Pan
 *
 * Copyright 2024 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HDF5UTILS_HPP_
#define HDF5UTILS_HPP_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <hdf5.h>

#include "HDF5Types.hpp"
#include "mxx/datatypes.hpp"

struct HDF5PathData {
    const std::string& file_name;
    const std::string& root_path;
    const std::string& matrix_data_path;
    const std::string& var_data_path;
    const std::string& obs_data_path;
};

template <typename DataType> struct HDF5Utils {
    //
    using ssize_pair_t = std::pair<ssize_t, ssize_t>;
    using string_vector_t = std::vector<std::string>;
    //
    static const ssize_t MAX_DIMS = 2;
    //
    static inline size_t strnlen(char* str, size_t n) {
        size_t i = 0;
        while ((i < n) && (str[i] != 0))
            ++i;
        return i;
    }

    //
    static inline size_t get_cacheline_size() {
        int64_t val = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        if (val < 1) {
            std::cerr
                << "Warning : " << __FILE__ << ":" << __LINE__
                << " Unable to detect cacheline size.  using default of 64"
                << std::endl;
            return 64UL;
        } else
            return val;
    }

    static inline size_t get_aligned_size(size_t const& bytes,
                                          size_t const& align) {
        if ((align & (align - 1UL)) != 0)
            throw std::domain_error("Alignment is not a power of 2.\n");
        return (bytes + align - 1UL) & ~(align - 1UL);
    }

    static inline size_t get_aligned_size(size_t const& bytes) {
        return get_aligned_size(bytes, get_cacheline_size());
    }

    static std::optional<ssize_pair_t>
    get_data_size_2d(hid_t loc_id, const std::string& dset_name) {
        // From the answer https://stackoverflow.com/a/15791325
        //   - for the question  https://stackoverflow.com/q/15786626
        auto exists = H5Lexists(loc_id, dset_name.c_str(), H5P_DEFAULT);
        if (exists <= 0) {
            return std::nullopt;
        }
        //
        hid_t dataset_id = H5Dopen(loc_id, dset_name.c_str(), H5P_DEFAULT);
        hid_t dataspace_id = H5Dget_space(dataset_id);
        int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        assert(ndims <= MAX_DIMS);
        //
        hsize_t dims[MAX_DIMS];
        H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        return std::make_pair(dims[0], dims[1]);
    }

    static std::optional<ssize_pair_t>
    get_data_size_2d(const std::string& file_name,
                     const std::string& group_name,
                     const std::string& dset_name) {
        // open the file for reading only.
        hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0)
            return std::nullopt;

        hid_t group_id;
        auto status = H5Lexists(file_id, group_name.c_str(), H5P_DEFAULT);
        if (status > 0) {
            group_id = H5Gopen(file_id, group_name.c_str(), H5P_DEFAULT);
        } else {
            std::cerr << "WARN: unable to get group " << group_name
                      << " in file " << file_name << std::endl;
            H5Fclose(file_id);
            return std::nullopt;
        }

        std::optional<ssize_pair_t> res = get_data_size_2d(group_id, dset_name);
        H5Gclose(group_id);
        H5Fclose(file_id);
        return res;
    }

    // get gene expression matrix size
    static std::optional<ssize_pair_t>
    get_matrix_size(const std::string& file_name, const std::string& group_name,
                    const std::string& mtx_dset_name) {
        return get_data_size_2d(file_name, group_name, mtx_dset_name);
    }

    static std::optional<ssize_pair_t>
    get_matrix_size(const std::string& file_name, const std::string& group_name,
                    const std::string& mtx_dset_name, MPI_Comm comm) {
        int rank, procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);
        if (procs == 1) {
            return get_data_size_2d(file_name, group_name, mtx_dset_name);
        }
        //
        std::optional<ssize_pair_t> res = std::make_pair(0, 0);
        if (rank == 0) {
            res = get_data_size_2d(file_name, group_name, mtx_dset_name);
            if (!res.has_value()) {
                res = std::make_pair(0, 0);
            }
        }
        mxx::datatype dtx = mxx::get_datatype<ssize_t>();
        // splash::utils::mpi::datatype<ssize_t> size_type;
        MPI_Bcast(&(res->first), 1, dtx.type(), 0, comm);
        MPI_Bcast(&(res->second), 1, dtx.type(), 0, comm);
        return res;
    }

    static std::optional<string_vector_t>
    read_strings(hid_t file_id, std::string const& dset_name) {
        // from https://stackoverflow.com/questions/581209
        // open data set
        auto exists = H5Lexists(file_id, dset_name.c_str(), H5P_DEFAULT);
        if (exists <= 0) {
            return std::nullopt;
        }
        hid_t dataset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);
        hid_t filetype_id = H5Dget_type(dataset_id);
        if (H5Tdetect_class(filetype_id, H5T_STRING) <= 0) {
            std::cerr << "ERROR: NOT a string type dataset " << dset_name
                      << std::endl;
            H5Dclose(dataset_id);
            H5Tclose(filetype_id);
            return {};
        }

        size_t max_len = H5Tget_size(filetype_id);
        // auto cid =  H5Tget_class (filetype_id);
        // FMT_ROOT_PRINT("In read dataset, cid [{}];  ptr size {}\n",
        //               cid == H5T_STRING ? "string" : "non-string",  max_len
        //               );
        // open data space and get dimensions
        hid_t dataspace_id = H5Dget_space(dataset_id);
        assert(H5Sget_simple_extent_ndims(dataspace_id) <= MAX_DIMS);
        hsize_t dims[MAX_DIMS];
        H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
        string_vector_t out;
        // FMT_ROOT_PRINT("In read STRING dataset, got number of strings: [{}]
        // \n", dims[0]);
        if (H5Tis_variable_str(filetype_id)) {
            // Variable length string type -- max_len is the ptr size
            // data is array of pointers : char **
            // https://docs.hdfgroup.org/hdf5/v1_14/group___h5_t.html#title25
            char** data = reinterpret_cast<char**>(
                calloc(dims[0], max_len * sizeof(char)));

            // prepare output
            // auto status =
            H5Dread(dataset_id, filetype_id, H5S_ALL, dataspace_id, H5P_DEFAULT,
                    data);

            // convert to string objects
            out.clear();
            out.reserve(dims[0]);
            char* ptr = *data;
            for (size_t x = 0; x < dims[0]; ++x, ptr = *(data + x)) {
                auto l = strlen(ptr);
                // FMT_ROOT_PRINT("GOT STRING {} {:p} {} \"{}\"\n", x, ptr, l,
                // std::string(ptr, l) );
                out.emplace_back(ptr, l);
            }
            // TODO(x): vlen reclaim is supposed to be deprecated
            //   libraries in current server doesn't reflect this
            // https://docs.hdfgroup.org/hdf5/v1_12/group___h5_d.html#title31
            H5Dvlen_reclaim(filetype_id, dataspace_id, H5P_DEFAULT, data);
            free(data);
        } else {
            // Fixed length strings :  max_len is the size of the string;
            // data should be a continuous memory block.
            char* data = reinterpret_cast<char*>(
                calloc((dims[0] + 1), max_len * sizeof(char)));
            data[dims[0] * max_len] = 0;
            out.clear();
            out.reserve(dims[0]);

            // read the block of data
            H5Dread(dataset_id, filetype_id, H5S_ALL, dataspace_id, H5P_DEFAULT,
                    data);
            //
            char* ptr = data;
            for (size_t x = 0; x < dims[0]; ++x, ptr += max_len) {
                // auto l = strlen(ptr);
                // FMT_ROOT_PRINT("GOT STRING {} {:p} {} \"{}\"\n", x, ptr, l,
                // std::string(ptr, l) );
                out.emplace_back(ptr, strnlen(ptr, max_len));
            }
            free(data);
        }
        // std::cout << " OUT " << out.size() << " " << dims[0] << std::endl;

        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Tclose(filetype_id);

        return out;
    }

    // specify number of rows and cols to read.
    static bool read_matrix(hid_t file_id, const std::string& dset_name,
                            const size_t rows, const size_t cols,
                            DataType *mtx_data) {
        // open data set
        auto exists = H5Lexists(file_id, dset_name.c_str(), H5P_DEFAULT);
        if (exists <= 0) {
            return false;
        }

        hid_t dataset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);

        // open data space and get dimensions
        hid_t dataspace_id = H5Dget_space(dataset_id);
        int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        assert(ndims <= MAX_DIMS);
        hsize_t file_dims[MAX_DIMS];
        H5Sget_simple_extent_dims(dataspace_id, file_dims, NULL);

        // create target space
        hsize_t mem_dims[2] = {rows, cols};
        hid_t memspace_id = H5Screate_simple(ndims, mem_dims, NULL);
        // select hyperslab of memory, for row by row traversal
        hsize_t mstart[2] = {0, 0};      // element offset for first block
        hsize_t mcount[2] = {rows, 1};   // # of blocks
        hsize_t mstride[2] = {1, cols};  // element stride to get to next block
        hsize_t mblock[2] = {1, cols};   // block size  1xcols
        H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, mstart, mstride,
                            mcount, mblock);
        // data type
        splash::utils::hdf5::datatype<DataType> htype;
        hid_t type_id = htype.value;

        // read data.
        // auto status =
        H5Dread(dataset_id, type_id, memspace_id, dataspace_id, H5P_DEFAULT,
                mtx_data);

        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        return true;
    }

    static bool read_matrix(hid_t file_id, std::string const& dset_name,
                            size_t rows, size_t cols, MPI_Comm comm,
                            DataType* mat_data) {
        int procs, rank;
        MPI_Comm_size(comm, &procs);
        MPI_Comm_rank(comm, &rank);

        size_t row_offset = rows;
        MPI_Exscan(MPI_IN_PLACE, &row_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                   comm);
        if (rank == 0)
            row_offset = 0;

        // open data set
        auto exists = H5Lexists(file_id, dset_name.c_str(), H5P_DEFAULT);
        if (exists <= 0)
            return false;

        hid_t dataset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);

        // open data space and get dimensions
        hid_t filespace_id = H5Dget_space(dataset_id);
        int ndims = H5Sget_simple_extent_ndims(filespace_id);
        assert(ndims <= MAX_DIMS);
        hsize_t file_dims[MAX_DIMS];
        H5Sget_simple_extent_dims(filespace_id, file_dims, NULL);

        // each process defines dataset in memory and hyperslab in file.
        hsize_t start[2] = {row_offset, 0};  // starting offset, row, then col.
        hsize_t count[2] = {rows, cols};     // number of row and col blocks.
        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start, NULL, count,
                            NULL);

        hsize_t mem_dims[2] = {rows, cols};
        hid_t memspace_id = H5Screate_simple(ndims, mem_dims, NULL);
        // select hyperslab of memory, for row by row traversal
        hsize_t mstart[2] = {0, 0};      // element offset for first block
        hsize_t mcount[2] = {rows, 1};   // # of blocks
        hsize_t mstride[2] = {1, cols};  // element stride to get to next block
        hsize_t mblock[2] = {1, cols};   // block size  1xcols
        H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, mstart, mstride,
                            mcount, mblock);

        // float type
        splash::utils::hdf5::datatype<DataType> type;
        hid_t type_id = type.value;

        // read data.  use mem_type is variable length string.  memspace is same
        // as file space.
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        // auto status =
        H5Dread(dataset_id, type_id, memspace_id, filespace_id, plist_id,
                mat_data);

        // convert to string objects
        H5Sclose(memspace_id);
        H5Sclose(filespace_id);
        H5Dclose(dataset_id);
        H5Pclose(plist_id);
        return true;
    }

};

#endif
