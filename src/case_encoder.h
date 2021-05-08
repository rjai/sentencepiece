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
constexpr char cAllUppercase = 'A';
constexpr char cTitlecase = 'T';
constexpr char cLowercase = 'L';

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

  static inline bool isUpper(const absl::string_view& sp) { return sp[0] == cUppercase; }
  static inline bool isLower(const absl::string_view& sp) { return sp[0] == cLowercase; }
  static inline bool isNeutral(const absl::string_view& sp) { return !isLower(sp) && !isUpper(sp); }
};

class UpperCaseEncoder : public CaseEncoder {
private:
  std::string buffer_;
  
  bool upperCaseMode_{false};
  bool prevNeutral_{false};
  size_t numUpper_ = 0;
  size_t numSpans_ = 0;
  bool flushed_{false};

  absl::string_view buffer(absl::string_view sp, int rm, bool last) {
    if(last) {
      return flush(sp, rm);
    } else {
      flushed_ = false;
      sp.remove_prefix(rm);
      buffer_.append(sp.data(), sp.size());
      return absl::string_view(nullptr, 0);
    }
  }

  absl::string_view flush(absl::string_view sp, int rm) {
    flushed_ = true;
    if(rm)
      sp.remove_prefix(rm);

    if(!buffer_.empty()) {
      buffer_.append(sp.data(), sp.size());
      return absl::string_view(buffer_);
    } else {
      return sp;
    }
  }

public:
  UpperCaseEncoder() {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    //std::cerr << "IN:  " << std::string(input) << std::endl;

    auto p = CaseEncoder::normalizePrefix(input);

    //std::cerr << "|" << std::string(p.first) << "| " << p.second << std::endl;

    auto sp = p.first;
    int consumed = p.second;

    bool last = input.size() == (size_t)consumed;
    decltype(p) ret = p;

    if(flushed_)
      buffer_.clear();

    if(isNeutral(sp)) {
    
      ret.first = flush(sp, 0);
    
    } else if(isLower(sp)) {
    
      if(numUpper_ == 0) { // only letter
        ret.first = flush(sp, 1);
      } else if(numUpper_ == 1) { // convert to T
        ret.first = flush(sp, 1);
        const_cast<char&>(ret.first[0]) = cTitlecase;
      } else /*if(numUpper_ > 1)*/ { // with L
        ret.first = flush(sp, 0);
      }
      numUpper_ = 0;
    
    } else if(isUpper(sp)) {
    
      if(numUpper_ == 0)
        ret.first = buffer(sp, 0, last);
      else if(numUpper_ > 0)
        ret.first = flush(sp, 1);
      numUpper_++;
    
    }

    // if(state_ == 0 || state_ == 4)
    //   buffer_.clear();

    // if(isUpper(sp)) {
    //   if(state_ == 0) {
    //     buffer_.append(sp.data(), sp.size());
    //     buffer_[0] = cTitlecase;
    //     state_ = 1;
    //     ret = {{nullptr, 0}, consumed};
    //   } else if(state_ == 1 || state_ == 2) {
    //     buffer_.append(sp.data() + 1, sp.size() - 1);
    //     buffer_[0] = cUppercase;
    //     state_ = 2;
    //     ret = {{nullptr, 0}, consumed};
    //   } else if(state_ == 3) {
    //     buffer_.append(sp.data() + 1, sp.size() - 1);
    //     buffer_[0] = cAllUppercase;
    //     state_ = 4;
    //     p.first = absl::string_view(buffer_);
    //     ret = p;
    //   } else if(state_ == 4) {
    //     p.first.remove_prefix(1);
    //     if(!buffer_.empty()) {
    //       buffer_.append(p.first.data(), p.first.size());
    //       p.first = absl::string_view(buffer_);
    //     }
    //     state_ = 4;
    //     ret = p;
    //   }
    //   if(last)
    //     ret.first = absl::string_view(buffer_);
    // } else if(isLower(sp)) {
    //   p.first.remove_prefix(1);
    
    //   if(state_ == 2 || state_ == 4)
    //     buffer_.append(1, cLowercase);
    
    //   if(!buffer_.empty()) {
    //     buffer_.append(p.first.data(), p.first.size());
    //     p.first = absl::string_view(buffer_);
    //   }
    //   state_ = 0;
    //   ret = p;
    // } else /*if(isNeutral(sp))*/ {
    //   if(!buffer_.empty()) {
    //     buffer_.append(p.first.data(), p.first.size());
    //     p.first = absl::string_view(buffer_);
    //   }
    //   if(state_ == 2) {
    //     state_ = 3;
    //     ret = {{nullptr, 0}, consumed};
    //   } else {
    //     ret = p;
    //   }
    // }

    // std::cerr << upperCaseMode_ << " " << numUpper_ << " |" << std::string(ret.first) << "| " << ret.second << std::endl;
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
