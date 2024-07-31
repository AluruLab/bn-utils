/**
 * @file HDF5DataReader.hpp
 * @brief Declaration of the functions for reading files.
 * @author Ankit Srivastava <asrivast@gatech.edu>
 *
 * Copyright 2020 Georgia Institute of Technology
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
#ifndef HDF5_DATAREADER_HPP_
#define HDF5_DATAREADER_HPP_

#include <string>
#include <vector>

#include "DataReader.hpp"

#include "HDF5Utils.hpp"
#include <hdf5.h>

/**
 * @brief Class that reads a file with observations in HDF5 file.
 *
 * @tparam DataType The type of the data to be read.
 */
template <typename DataType>
class HDF5ObservationReader : public DataReader<DataType> {
    std::vector<std::string> m_obsNames;

  public:
    HDF5ObservationReader(const std::string&, const uint32_t, const uint32_t,
                          const std::string&, const std::string&,
                          const std::string&, const std::string&,
                          const bool = false);
};  // class RowObservationReader

template <typename DataType>
/**
 * @brief Constructor that reads data from the file.
 *
 * @param fileName The name of the file to be read.
 * @param numRows Total number of rows (variables) in the file.
 * @param numCols Total number of columns (observations) in the file.
 * @param path Path in HDF5 file within which all data is stored
 * @param matrixPath Path in HDF5 file within which the matrix is stored
 * @param obsPath Path in HDF5 file within which the observation names are
 * stored
 * @param varsPath Path in HDF5 file within which the variables names are stored
 * @param parallelRead If the data should be read in parallel.
 */
HDF5ObservationReader<DataType>::HDF5ObservationReader(
    const std::string& fileName, const uint32_t numRows, const uint32_t numCols,
    const std::string& path, const std::string& matrixPath,
    const std::string& obsPath, const std::string& varsPath,
    const bool parallelRead)
    : DataReader<DataType>(numRows, false, true) {

    using HDF5Ifx = HDF5Utils<DataType>;
    mxx::comm comm;

    //
    // ssize_t stride_bytes =
    //    HDF5Ifx::get_aligned_size(numCols * sizeof(DataType), 0);
    if (parallelRead && comm.size() > 1) {
        // read the names.
        MPI_Info info = MPI_INFO_NULL;
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, info);

        hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, plist_id);
        if (file_id < 0) {
            std::cerr << "ERROR: failed to open PHDF5 file " << fileName
                      << std::endl;
            H5Pclose(plist_id);
            throw std::runtime_error("Failed to Open File : " + fileName);
        }
        hid_t group_id;
        auto status = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
        if (status > 0) {
            group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
        } else {
            std::cerr << "WARN: unable to get group " << path << " in file "
                      << fileName << std::endl;
            H5Fclose(file_id);
            H5Pclose(plist_id);
            throw std::runtime_error("WARN: unable to get group " + path +
                                     " in file " + fileName);
        }

        try {
            this->m_varNames =
                HDF5Ifx::read_strings(group_id, varsPath).value();
            m_obsNames = HDF5Ifx::read_strings(group_id, obsPath).value();
        } catch (const std::bad_optional_access& e) {
            throw std::runtime_error(
                " Failed to load Variable and observation names from " +
                varsPath + " and " + obsPath);
        }
        // read the data.
        HDF5Ifx::read_matrix(group_id, matrixPath, numRows, numCols, comm,
                             this->m_data.data());
        H5Gclose(group_id);
        H5Fclose(file_id);
        H5Pclose(plist_id);
    } else {
        this->m_data.resize(numRows * numCols);
        // open the file for reading only.
        hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            throw std::runtime_error("Failed to Open File : " + fileName);
        }

        hid_t group_id;
        auto status = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
        if (status > 0) {
            group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
        } else {
            std::cout << "WARN: unable to get group " << path << " in file "
                      << fileName << std::endl;
            H5Fclose(file_id);
            throw std::runtime_error("WARN: unable to get group " + path +
                                     " in file " + fileName);
        }

        // read the names.
        try {
            this->m_varNames =
                HDF5Ifx::read_strings(group_id, varsPath).value();
            m_obsNames = HDF5Ifx::read_strings(group_id, obsPath).value();
        } catch (const std::bad_optional_access& e) {
            throw std::runtime_error(
                " Failed to load Variable and observation names from " +
                varsPath + " and " + obsPath);
        }

        // read the data.
        HDF5Ifx::read_matrix(group_id, matrixPath, numRows, numCols,
                             this->m_data.data());
        H5Gclose(group_id);
        H5Fclose(file_id);
    }
}

#endif
