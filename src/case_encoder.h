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

class CaseEncoder {
protected:
  typedef std::function<std::pair<absl::string_view, int>(absl::string_view)> Normalizer;
  Normalizer normalizer_;

public:
  virtual ~CaseEncoder() {}
  
  virtual std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    return normalizer_(input);
  }

  virtual void push(const std::pair<absl::string_view, int>& p, bool last) = 0;
  virtual bool empty() = 0;
  virtual std::pair<absl::string_view, int> pop() = 0;

  static std::unique_ptr<CaseEncoder> Create(bool, bool);

  virtual void setNormalizer(Normalizer normalizer) {
    normalizer_ = normalizer;
  }
};

class IdentityCaseEncoder : public CaseEncoder {
private:
  std::pair<absl::string_view, int> p_;
  bool empty_{true};

public:
  IdentityCaseEncoder() {}
  
  void push(const std::pair<absl::string_view, int>& p, bool /*last*/) {
    p_ = p;
    empty_ = false;
  }

  bool empty() {
    return empty_;
  }

  std::pair<absl::string_view, int> pop() {
    empty_ = true;
    return p_;
  }
};

class UpperCaseEncoder : public CaseEncoder {
  std::vector<std::string> buffers_;
  std::deque<std::pair<absl::string_view, int>> pieces_;
  bool flush_{false};
  size_t countU_{0};

  void fixUs() {
    if(countU_ == 1) {
      auto sp = pieces_.front().first;
      buffers_.emplace_back(sp.data(), sp.size());
      buffers_.back()[0] = 'T';
      pieces_.front().first = absl::string_view(buffers_.back());
    } else if(countU_ > 1) {
      for(int i = 1; i < countU_; ++i) {
        auto sp = pieces_[i].first;
        pieces_[i].first = absl::string_view(sp.data() + 1, sp.size() - 1);
      }
    }
  }

public:
  UpperCaseEncoder() {}

  void push(const std::pair<absl::string_view, int>& p, bool last) {
    auto sp = p.first;
    if(sp.data()[0] == 'U') {
      pieces_.push_back(p);
      countU_++;
      flush_ = false;
    } else if(sp.data()[0] == 'P') {
      fixUs();
      pieces_.push_back({absl::string_view(sp.data() + 1, sp.size() - 1), p.second});
      countU_ = 0;
      flush_ = true;
    } else if(sp.data()[0] == ' ') {
      fixUs();
      pieces_.push_back(p);
      countU_ = 0;
      flush_ = true;
    } else {
      fixUs();
      if(countU_ > 1) {
        buffers_.emplace_back("L");
        buffers_.back().append(p.first.data(), p.first.size());
        pieces_.push_back({buffers_.back(), p.second});
      } else {
        pieces_.push_back(p);
      }
      countU_ = 0;
      flush_ = true;
    }

    if(last)
      flush_ = true; // flush it all out
  }

  bool empty() {
    return pieces_.empty() || !flush_;
  }

  std::pair<absl::string_view, int> pop() {
    auto p = pieces_.front();
    pieces_.pop_front();
    return p;
  }
};

class UpperCaseDecoder : public CaseEncoder {
private:
  std::unique_ptr<std::string> buffer_;
  absl::string_view input_;

  std::pair<absl::string_view, int> p_;
  bool empty_{true};

  int state_ = 0;

public:
  UpperCaseDecoder() {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    if(!buffer_) {
      buffer_.reset(new std::string(input.data(), input.size()));
      input_ = absl::string_view(*buffer_);
    }

    auto p = CaseEncoder::normalizePrefix(input_);

    if(input_[0] == 'U') {
      if(state_ == 0) { 
        input_.remove_prefix(p.second - 1);
        const_cast<char&>(input_[0]) = 'U';
        state_ = 1;
      } else if(state_ == 1) {
        if(p.second > 1) {
          input_.remove_prefix(p.second - 1);
          const_cast<char&>(input_[0]) = 'U';
          p.second = p.second - 1;
          state_ = 1;
        } else {
          input_.remove_prefix(p.second);
          p.first.remove_prefix(1);
          p.second = 0;
          state_ = 0;
        }
      }
    } else if(input_[0] == 'L') {
      input_.remove_prefix(p.second);
      p.first.remove_prefix(1);
      state_ = 0;
    } else {
      input_.remove_prefix(p.second);
      state_ = 0;
    }

    return p;
  }

  void push(const std::pair<absl::string_view, int>& p, bool /*last*/) {
    p_ = p;
    empty_ = false;
  }

  bool empty() {
    return empty_;
  }

  std::pair<absl::string_view, int> pop() {
    empty_ = true;
    return p_;
  }
};

std::unique_ptr<CaseEncoder> CaseEncoder::Create(bool encodeCase, bool decodeCase) {
  if(encodeCase && decodeCase) {
    LOG(ERROR) << "Cannot set both encodeCase=true and decodeCase=true";
    return nullptr;
  } else if(encodeCase) {
    return std::unique_ptr<CaseEncoder>(new UpperCaseEncoder());
  } else if(decodeCase) {
    return std::unique_ptr<CaseEncoder>(new UpperCaseDecoder());
  } else {
    return std::unique_ptr<CaseEncoder>(new IdentityCaseEncoder());
  }
}

}  // namespace normalizer
}  // namespace sentencepiece
#endif  // NORMALIZER_CASE_ENCODER_H_
