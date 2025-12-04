#### C++
C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

dynamicemb uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
to check your C/C++ changes. Sometimes you have some manually formatted
code that you don’t want clang-format to touch.
You can disable formatting like this:

```cpp
int formatted_code;
// clang-format off
    void    unformatted_code  ;
// clang-format on
void formatted_code_again;
```

Install Clang-format (the version 18.1.3 is required) for Ubuntu:

```bash
sudo apt install clang-format-18
```

format all with:
```bash
find ./src -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) -exec clang-format-18 -i {} \;

```

### Python
dynamicemb formats the Python code using pre-commit.

Install pre-commit using pip:

```bash
pip install pre-commit
```

Format the code manually:
```bash
pre-commit run --all-files
```
See more details at file .pre-commit-config.yaml
