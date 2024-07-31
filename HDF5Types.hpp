/*
 * Copyright 2021 Georgia Tech Research Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): Tony C. Pan
 */

#pragma once

#include <bitset>
#include <type_traits>
#include <hdf5.h>

namespace splash { namespace utils { namespace hdf5 {



template <typename T, bool BUILTIN = std::is_arithmetic<T>::value>
struct datatype;

// see https://support.hdfgroup.org/HDF5/doc/RM/PredefDTypes.html

// template specialized structs.
template <typename TT> struct datatype<TT*, false> {  hid_t value { H5T_NATIVE_HADDR }; };

// duplicate uint64_t and int64_t
// template <> struct datatype<size_t, true> {  hid_t value { H5T_NATIVE_HSIZE }; };
// template <> struct datatype<ssize_t, true> {  hid_t value { H5T_NATIVE_HSSIZE }; };

template <> struct datatype<bool, true> {  hid_t value { H5T_NATIVE_HBOOL }; };

template <> struct datatype<char, true> {  hid_t value { H5T_NATIVE_CHAR }; };
template <> struct datatype<signed char, true> {  hid_t value { H5T_NATIVE_SCHAR }; };
template <> struct datatype<unsigned char, true> {  hid_t value { H5T_NATIVE_UCHAR }; };

template <> struct datatype<wchar_t, true> {  hid_t value { H5T_NATIVE_SHORT }; };
template <> struct datatype<short, true> {  hid_t value { H5T_NATIVE_SHORT }; };
template <> struct datatype<unsigned short, true> {  hid_t value { H5T_NATIVE_USHORT }; };

template <> struct datatype<int, true> {  hid_t value { H5T_NATIVE_INT }; };
template <> struct datatype<unsigned int, true> {  hid_t value { H5T_NATIVE_UINT}; };

template <> struct datatype<long, true> {  hid_t value { H5T_NATIVE_LONG }; };
template <> struct datatype<unsigned long, true> {  hid_t value { H5T_NATIVE_ULONG }; };
template <> struct datatype<long long, true> {  hid_t value { H5T_NATIVE_LLONG }; };
template <> struct datatype<unsigned long long, true> {  hid_t value { H5T_NATIVE_ULLONG }; };

template <> struct datatype<float, true> {  hid_t value { H5T_NATIVE_FLOAT }; };
template <> struct datatype<double, true> {  hid_t value { H5T_NATIVE_DOUBLE }; };
template <> struct datatype<long double, true> {  hid_t value { H5T_NATIVE_LDOUBLE }; };

template <> struct datatype<std::bitset<8>, false> { hid_t value { H5T_NATIVE_B8 }; };
template <> struct datatype<std::bitset<16>, false> { hid_t value { H5T_NATIVE_B16 }; };
template <> struct datatype<std::bitset<32>, false> { hid_t value { H5T_NATIVE_B32 }; };
template <> struct datatype<std::bitset<64>, false> { hid_t value { H5T_NATIVE_B64 }; };

}}}
