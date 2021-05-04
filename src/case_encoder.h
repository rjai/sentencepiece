// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#ifndef NORMALIZER_CASE_ENCODER_H_
#define NORMALIZER_CASE_ENCODER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "third_party/absl/strings/string_view.h"


namespace sentencepiece {
namespace normalizer {

class CaseEncoder {
public:
  virtual ~CaseEncoder() {}
  virtual bool encode(const absl::string_view& sp, int n, int consumed) = 0;

  static std::unique_ptr<CaseEncoder> Create(bool, bool, std::string* /*normalized*/, std::vector<size_t>* /*norm_to_orig*/);
};

class IdentityCaseEncoder : public CaseEncoder {
public:
  IdentityCaseEncoder() {}
  bool encode(const absl::string_view& sp, int n, int consumed) {
    return true;
  }
};

class UpperCaseEncoder : public CaseEncoder {
private:
  char* last_u_{nullptr};
  size_t last_u_dist_{0};
  std::string* normalized_;
  std::vector<size_t> *norm_to_orig_;

public:
  UpperCaseEncoder(std::string* normalized, std::vector<size_t> *norm_to_orig)
  : normalized_(normalized), norm_to_orig_(norm_to_orig) {}

  bool encode(const absl::string_view& sp, int n, int consumed) {
    if(n != 0)
      return true;

    char curChar = sp.data()[0];
    if(curChar == ' ') {
      if(last_u_) {
        if(last_u_dist_ == 1)
          *last_u_ = 'T';
        last_u_ = nullptr;
        last_u_dist_ = 0;
      }
      return true;
    }

    if(!last_u_ && curChar == 'U') {
      last_u_ = &(*normalized_)[0] + normalized_->size();
      last_u_dist_++;
    } else if(last_u_ && curChar == 'U') { // uppercase sequence, skip over U
      last_u_dist_++;
      return false;
    } else if (last_u_ && curChar != 'U' && last_u_dist_ == 1) { // single uppercase letter
      *last_u_ = 'T';
      last_u_ = nullptr;
      last_u_dist_ = 0;
    } else if (last_u_ && (curChar != 'U' && curChar != 'P') && last_u_dist_ > 1) { 
      // we had a longer uppercase sequence, hence need to insert 'L'
      normalized_->append(1, 'L');
      norm_to_orig_->push_back(consumed); 

      last_u_ = nullptr;
      last_u_dist_ = 0;
    } else {
      last_u_ = nullptr;
      last_u_dist_ = 0;
    }

    if(curChar == 'P') {
      last_u_ = nullptr;
      last_u_dist_ = 0;
      return false;
    }

    return true;
  }
};

std::unique_ptr<CaseEncoder> CaseEncoder::Create(bool encodeCase, bool decodeCase, std::string* normalized, std::vector<size_t>* norm_to_orig) {
  if(encodeCase && decodeCase) {
    LOG(ERROR) << "Cannot set both encodeCase=true and decodeCase=true";
    return nullptr;
  } else if(encodeCase) {
    return std::unique_ptr<CaseEncoder>(new UpperCaseEncoder(normalized, norm_to_orig));
  } else if(decodeCase) {
    LOG(ERROR) << "Not implemented";
    return nullptr;
  } else {
    return std::unique_ptr<CaseEncoder>(new IdentityCaseEncoder());
  }
}

}  // namespace normalizer
}  // namespace sentencepiece
#endif  // NORMALIZER_CASE_ENCODER_H_
