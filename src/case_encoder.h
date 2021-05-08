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
#include <deque>

#include "common.h"
#include "third_party/absl/strings/string_view.h"


namespace sentencepiece {
namespace normalizer {

constexpr char cUppercase = 'U';
constexpr char cTitlecase = 'T';
constexpr char cLowercase = 'L';
constexpr char cPunctuation = 'P';
constexpr char cSpace = ' ';

class CaseEncoder {
protected:
  typedef std::function<std::pair<absl::string_view, int>(absl::string_view)> Normalizer;
  Normalizer normalizer_;

public:
  virtual ~CaseEncoder() {}
  
  virtual std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    return normalizer_(input);
  }

  virtual void setNormalizer(Normalizer normalizer) {
    normalizer_ = normalizer;
  }

  static std::unique_ptr<CaseEncoder> Create(bool, bool);
};

class UpperCaseEncoder : public CaseEncoder {
private:
  std::string buffer_;
  int state_ = 0;

public:
  UpperCaseEncoder() {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    auto p = CaseEncoder::normalizePrefix(input);
    auto sp = p.first;
    int consumed = p.second;

    bool last = input.size() == (size_t)consumed;
    decltype(p) ret;

    auto null = [](int consumed) -> std::pair<absl::string_view, int> {
      return {{nullptr, 0}, consumed};
    };

    auto buffer = [this](absl::string_view sp) {
      buffer_.append(sp.data(), sp.size());
    };

    auto isUpper  = [=](absl::string_view sp) { return sp[0] == cUppercase;   };
    auto isSpace  = [=](absl::string_view sp) { return sp[0] == ' '; };

    if(state_ == 0)
      buffer_.clear();

    if(isUpper(sp)) {
      if(state_ == 0) {
        buffer(sp);
        
        buffer_[0] = cTitlecase;
        state_ = 1;
        ret = null(consumed);
      } else if(state_ == 1 || state_ == 2) {
        sp.remove_prefix(1);
        buffer(sp);

        buffer_[0] = cUppercase;
        state_ = 2;
        ret = null(consumed);
      }  

      if(last)
        ret.first = absl::string_view(buffer_);

    } else {
      if(state_ == 2 && !isSpace(sp))
        buffer_ += cLowercase;

      if(!buffer_.empty()) {
        buffer(sp);
        p.first = absl::string_view(buffer_);
      } else {
        p.first = sp;
      }
      
      state_ = 0;
      ret = p;
    }

    return ret;
  }
};

class UpperCaseDecoder : public CaseEncoder {
private:
  std::unique_ptr<std::string> buffer_;
  absl::string_view input_;

  int state_ = 0;

public:
  UpperCaseDecoder() {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    if(!buffer_) {
      buffer_.reset(new std::string(input.data(), input.size()));
      input_ = absl::string_view(*buffer_);
    }

    auto p = CaseEncoder::normalizePrefix(input_);
    int consumed = p.second;

    if(input_[0] == cUppercase) {
      if(state_ == 0) { 
        input_.remove_prefix(consumed - 1);
        const_cast<char&>(input_[0]) = cUppercase;
        state_ = 1;
      } else if(state_ == 1) {
        if(consumed > 1) {
          input_.remove_prefix(consumed - 1);
          const_cast<char&>(input_[0]) = cUppercase;
          p.second = consumed - 1;
          state_ = 1;
        } else {
          input_.remove_prefix(consumed);
          p.first.remove_prefix(1);
          p.second = 0;
          state_ = 0;
        }
      }
    } else if(input_[0] == cLowercase) {
      input_.remove_prefix(consumed);
      p.first.remove_prefix(1);
      state_ = 0;
    } else {
      input_.remove_prefix(consumed);
      state_ = 0;
    }

    return p;
  }
};

}  // namespace normalizer
}  // namespace sentencepiece
#endif  // NORMALIZER_CASE_ENCODER_H_
