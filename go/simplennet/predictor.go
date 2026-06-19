//go:build windows

// Package simplennet is a thin Go wrapper over the libsimplennet native DLL.
package simplennet

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"
)

// OutputType matches the native predictor output mode.
type OutputType int

const (
	OutputInt OutputType = iota
	OutputFloat
	OutputCategory
)

// SimplePredictor holds a native predictor handle and the DLL it came from.
type SimplePredictor struct {
	OutputType OutputType
	lib        *nativeLibrary
	handle     uintptr
}

// NewSimplePredictor creates a predictor using the default native library path.
func NewSimplePredictor(outputType OutputType) (*SimplePredictor, error) {
	lib, err := defaultLibrary()
	if err != nil {
		return nil, err
	}
	handle, err := lib.create(outputType)
	if err != nil {
		return nil, err
	}
	return &SimplePredictor{OutputType: outputType, lib: lib, handle: handle}, nil
}

// Close releases the native predictor handle.
func (p *SimplePredictor) Close() {
	if p != nil && p.lib != nil && p.handle != 0 {
		p.lib.destroy(p.handle)
		p.handle = 0
	}
}

// Fit trains a numeric predictor.
func (p *SimplePredictor) Fit(inputs [][]float64, targets []float64) error {
	if p.OutputType == OutputCategory {
		return errors.New("use FitLabels for categorical predictors")
	}
	flat, rows, cols, err := flattenMatrix(inputs)
	if err != nil {
		return err
	}
	if len(targets) != rows {
		return fmt.Errorf("target count %d does not match row count %d", len(targets), rows)
	}
	return p.lib.fitNumeric(p.handle, flat, rows, cols, targets, 30, 0.01)
}

// FitLabels trains a categorical predictor.
func (p *SimplePredictor) FitLabels(inputs [][]float64, targets []string) error {
	if p.OutputType != OutputCategory {
		return errors.New("use Fit for numeric predictors")
	}
	flat, rows, cols, err := flattenMatrix(inputs)
	if err != nil {
		return err
	}
	if len(targets) != rows {
		return fmt.Errorf("label count %d does not match row count %d", len(targets), rows)
	}
	return p.lib.fitLabels(p.handle, flat, rows, cols, targets, 30, 0.01)
}

// FitText trains a numeric predictor from serialized text or JSON inputs.
func (p *SimplePredictor) FitText(inputs []string, targets []float64) error {
	if p.OutputType == OutputCategory {
		return errors.New("use FitTextLabels for categorical predictors")
	}
	if len(inputs) == 0 {
		return errors.New("inputs must not be empty")
	}
	if len(targets) != len(inputs) {
		return fmt.Errorf("target count %d does not match input count %d", len(targets), len(inputs))
	}
	return p.lib.fitTextNumeric(p.handle, inputs, targets, 64, 30, 0.01)
}

// FitTextLabels trains a categorical predictor from serialized text or JSON inputs.
func (p *SimplePredictor) FitTextLabels(inputs []string, targets []string) error {
	if p.OutputType != OutputCategory {
		return errors.New("use FitText for numeric predictors")
	}
	if len(inputs) == 0 {
		return errors.New("inputs must not be empty")
	}
	if len(targets) != len(inputs) {
		return fmt.Errorf("label count %d does not match input count %d", len(targets), len(inputs))
	}
	return p.lib.fitTextLabels(p.handle, inputs, targets, 64, 30, 0.01)
}

// PredictNumbers predicts floating-point outputs.
func (p *SimplePredictor) PredictNumbers(inputs [][]float64) ([]float64, error) {
	flat, rows, cols, err := flattenMatrix(inputs)
	if err != nil {
		return nil, err
	}
	return p.lib.predictNumbers(p.handle, flat, rows, cols)
}

// PredictInts predicts integer outputs.
func (p *SimplePredictor) PredictInts(inputs [][]float64) ([]int, error) {
	flat, rows, cols, err := flattenMatrix(inputs)
	if err != nil {
		return nil, err
	}
	return p.lib.predictInts(p.handle, flat, rows, cols)
}

// PredictLabels predicts string labels.
func (p *SimplePredictor) PredictLabels(inputs [][]float64) ([]string, error) {
	if p.OutputType != OutputCategory {
		return nil, errors.New("predict labels requires a categorical predictor")
	}
	flat, rows, cols, err := flattenMatrix(inputs)
	if err != nil {
		return nil, err
	}
	return p.lib.predictLabels(p.handle, flat, rows, cols)
}

// PredictTextNumbers predicts floating-point outputs from serialized text or JSON inputs.
func (p *SimplePredictor) PredictTextNumbers(inputs []string) ([]float64, error) {
	if len(inputs) == 0 {
		return nil, errors.New("inputs must not be empty")
	}
	return p.lib.predictTextNumbers(p.handle, inputs)
}

// PredictTextInts predicts integer outputs from serialized text or JSON inputs.
func (p *SimplePredictor) PredictTextInts(inputs []string) ([]int, error) {
	if len(inputs) == 0 {
		return nil, errors.New("inputs must not be empty")
	}
	return p.lib.predictTextInts(p.handle, inputs)
}

// PredictTextLabels predicts string labels from serialized text or JSON inputs.
func (p *SimplePredictor) PredictTextLabels(inputs []string) ([]string, error) {
	if p.OutputType != OutputCategory {
		return nil, errors.New("predict labels requires a categorical predictor")
	}
	if len(inputs) == 0 {
		return nil, errors.New("inputs must not be empty")
	}
	return p.lib.predictTextLabels(p.handle, inputs)
}

// Save writes a .snet model directory.
func (p *SimplePredictor) Save(path string) error {
	return p.lib.save(p.handle, path)
}

// Load reads a .snet model directory.
func Load(path string) (*SimplePredictor, error) {
	lib, err := defaultLibrary()
	if err != nil {
		return nil, err
	}
	handle, err := lib.load(path)
	if err != nil {
		return nil, err
	}
	outputType, err := lib.outputType(handle)
	if err != nil {
		lib.destroy(handle)
		return nil, err
	}
	return &SimplePredictor{OutputType: outputType, lib: lib, handle: handle}, nil
}

func flattenMatrix(inputs [][]float64) ([]float64, int, int, error) {
	if len(inputs) == 0 {
		return nil, 0, 0, errors.New("inputs must not be empty")
	}
	cols := len(inputs[0])
	if cols == 0 {
		return nil, 0, 0, errors.New("inputs must contain at least one feature")
	}
	flat := make([]float64, 0, len(inputs)*cols)
	for _, row := range inputs {
		if len(row) != cols {
			return nil, 0, 0, errors.New("all input rows must have the same number of features")
		}
		flat = append(flat, row...)
	}
	return flat, len(inputs), cols, nil
}

type nativeLibrary struct {
	dll                  *syscall.LazyDLL
	createProc           *syscall.LazyProc
	destroyProc          *syscall.LazyProc
	fitNumericProc       *syscall.LazyProc
	fitLabelsProc        *syscall.LazyProc
	fitTextNumericProc   *syscall.LazyProc
	fitTextLabelsProc    *syscall.LazyProc
	predictNumericProc   *syscall.LazyProc
	predictIntProc       *syscall.LazyProc
	predictClassProc     *syscall.LazyProc
	predictTextProc      *syscall.LazyProc
	predictTextIntProc   *syscall.LazyProc
	predictTextClassProc *syscall.LazyProc
	classLabelCountProc  *syscall.LazyProc
	classLabelCopyProc   *syscall.LazyProc
	saveProc             *syscall.LazyProc
	loadProc             *syscall.LazyProc
	outputTypeProc       *syscall.LazyProc
}

func defaultLibrary() (*nativeLibrary, error) {
	path := os.Getenv("SIMPLENET_NATIVE_LIBRARY")
	if path == "" {
		candidates := []string{
			filepath.Join("..", "build-native", "Release", "simplennet.dll"),
			filepath.Join("build-native", "Release", "simplennet.dll"),
			"simplennet.dll",
			filepath.Join("..", "build-native", "Release", "simplennet_native.dll"),
			filepath.Join("build-native", "Release", "simplennet_native.dll"),
			"simplennet_native.dll",
		}
		for _, candidate := range candidates {
			if _, err := os.Stat(candidate); err == nil {
				path = candidate
				break
			}
		}
	}
	if path == "" {
		return nil, errors.New("set SIMPLENET_NATIVE_LIBRARY to the built simplennet.dll")
	}
	return loadLibrary(path)
}

func loadLibrary(path string) (*nativeLibrary, error) {
	if runtime.GOOS != "windows" {
		return nil, errors.New("native Go bindings are currently implemented for Windows DLLs")
	}

	lib := &nativeLibrary{dll: syscall.NewLazyDLL(path)}
	lib.createProc = lib.dll.NewProc("snet_create")
	lib.destroyProc = lib.dll.NewProc("snet_destroy")
	lib.fitNumericProc = lib.dll.NewProc("snet_fit_numeric_bits")
	lib.fitLabelsProc = lib.dll.NewProc("snet_fit_labels_bits")
	lib.fitTextNumericProc = lib.dll.NewProc("snet_fit_text_numeric_bits")
	lib.fitTextLabelsProc = lib.dll.NewProc("snet_fit_text_labels_bits")
	lib.predictNumericProc = lib.dll.NewProc("snet_predict_numeric")
	lib.predictIntProc = lib.dll.NewProc("snet_predict_ints")
	lib.predictClassProc = lib.dll.NewProc("snet_predict_class_indices")
	lib.predictTextProc = lib.dll.NewProc("snet_predict_text_numeric")
	lib.predictTextIntProc = lib.dll.NewProc("snet_predict_text_ints")
	lib.predictTextClassProc = lib.dll.NewProc("snet_predict_text_class_indices")
	lib.classLabelCountProc = lib.dll.NewProc("snet_class_label_count")
	lib.classLabelCopyProc = lib.dll.NewProc("snet_class_label_copy")
	lib.saveProc = lib.dll.NewProc("snet_save")
	lib.loadProc = lib.dll.NewProc("snet_load")
	lib.outputTypeProc = lib.dll.NewProc("snet_get_output_type")
	if err := lib.dll.Load(); err != nil {
		return nil, err
	}
	return lib, nil
}

func (lib *nativeLibrary) create(outputType OutputType) (uintptr, error) {
	var handle uintptr
	code, _, err := lib.createProc.Call(uintptr(outputType), uintptr(unsafe.Pointer(&handle)))
	if code != 0 {
		return 0, errorFromProc(err)
	}
	return handle, nil
}

func (lib *nativeLibrary) destroy(handle uintptr) {
	lib.destroyProc.Call(handle)
}

func (lib *nativeLibrary) fitNumeric(handle uintptr, flat []float64, rows, cols int, targets []float64, epochs int, lr float64) error {
	code, _, err := lib.fitNumericProc.Call(
		handle,
		flatPtr(flat),
		uintptr(rows),
		uintptr(cols),
		flatPtr(targets),
		uintptr(len(targets)),
		uintptr(epochs),
		uintptr(math.Float64bits(lr)),
	)
	if code != 0 {
		return errorFromProc(err)
	}
	return nil
}

func (lib *nativeLibrary) fitLabels(handle uintptr, flat []float64, rows, cols int, targets []string, epochs int, lr float64) error {
	cStrings, release, err := makeCStringArray(targets)
	if err != nil {
		return err
	}
	defer release()
	code, _, procErr := lib.fitLabelsProc.Call(
		handle,
		flatPtr(flat),
		uintptr(rows),
		uintptr(cols),
		uintptr(unsafe.Pointer(&cStrings[0])),
		uintptr(len(targets)),
		uintptr(epochs),
		uintptr(math.Float64bits(lr)),
	)
	if code != 0 {
		return errorFromProc(procErr)
	}
	return nil
}

func (lib *nativeLibrary) fitTextNumeric(handle uintptr, inputs []string, targets []float64, width int, epochs int, lr float64) error {
	cStrings, release, err := makeCStringArray(inputs)
	if err != nil {
		return err
	}
	defer release()
	code, _, procErr := lib.fitTextNumericProc.Call(
		handle,
		uintptr(unsafe.Pointer(&cStrings[0])),
		uintptr(len(inputs)),
		flatPtr(targets),
		uintptr(len(targets)),
		uintptr(width),
		uintptr(epochs),
		uintptr(math.Float64bits(lr)),
	)
	if code != 0 {
		return errorFromProc(procErr)
	}
	return nil
}

func (lib *nativeLibrary) fitTextLabels(handle uintptr, inputs []string, targets []string, width int, epochs int, lr float64) error {
	inputStrings, releaseInputs, err := makeCStringArray(inputs)
	if err != nil {
		return err
	}
	defer releaseInputs()
	targetStrings, releaseTargets, err := makeCStringArray(targets)
	if err != nil {
		return err
	}
	defer releaseTargets()
	code, _, procErr := lib.fitTextLabelsProc.Call(
		handle,
		uintptr(unsafe.Pointer(&inputStrings[0])),
		uintptr(len(inputs)),
		uintptr(unsafe.Pointer(&targetStrings[0])),
		uintptr(len(targets)),
		uintptr(width),
		uintptr(epochs),
		uintptr(math.Float64bits(lr)),
	)
	if code != 0 {
		return errorFromProc(procErr)
	}
	return nil
}

func (lib *nativeLibrary) predictNumbers(handle uintptr, flat []float64, rows, cols int) ([]float64, error) {
	outputs := make([]float64, rows)
	code, _, err := lib.predictNumericProc.Call(handle, flatPtr(flat), uintptr(rows), uintptr(cols), flatPtr(outputs), uintptr(len(outputs)))
	if code != 0 {
		return nil, errorFromProc(err)
	}
	return outputs, nil
}

func (lib *nativeLibrary) predictInts(handle uintptr, flat []float64, rows, cols int) ([]int, error) {
	outputs := make([]int64, rows)
	code, _, err := lib.predictIntProc.Call(handle, flatPtr(flat), uintptr(rows), uintptr(cols), uintptr(unsafe.Pointer(&outputs[0])), uintptr(len(outputs)))
	if code != 0 {
		return nil, errorFromProc(err)
	}
	results := make([]int, len(outputs))
	for i, value := range outputs {
		results[i] = int(value)
	}
	return results, nil
}

func (lib *nativeLibrary) predictLabels(handle uintptr, flat []float64, rows, cols int) ([]string, error) {
	indices := make([]uintptr, rows)
	code, _, err := lib.predictClassProc.Call(handle, flatPtr(flat), uintptr(rows), uintptr(cols), uintptr(unsafe.Pointer(&indices[0])), uintptr(len(indices)))
	if code != 0 {
		return nil, errorFromProc(err)
	}

	results := make([]string, len(indices))
	for i, index := range indices {
		buf := make([]byte, 256)
		code, _, err := lib.classLabelCopyProc.Call(handle, index, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
		if code != 0 {
			return nil, errorFromProc(err)
		}
		n := 0
		for n < len(buf) && buf[n] != 0 {
			n++
		}
		results[i] = string(buf[:n])
	}
	return results, nil
}

func (lib *nativeLibrary) predictTextNumbers(handle uintptr, inputs []string) ([]float64, error) {
	cStrings, release, err := makeCStringArray(inputs)
	if err != nil {
		return nil, err
	}
	defer release()
	outputs := make([]float64, len(inputs))
	code, _, procErr := lib.predictTextProc.Call(
		handle,
		uintptr(unsafe.Pointer(&cStrings[0])),
		uintptr(len(inputs)),
		flatPtr(outputs),
		uintptr(len(outputs)),
	)
	if code != 0 {
		return nil, errorFromProc(procErr)
	}
	return outputs, nil
}

func (lib *nativeLibrary) predictTextInts(handle uintptr, inputs []string) ([]int, error) {
	cStrings, release, err := makeCStringArray(inputs)
	if err != nil {
		return nil, err
	}
	defer release()
	outputs := make([]int64, len(inputs))
	code, _, procErr := lib.predictTextIntProc.Call(
		handle,
		uintptr(unsafe.Pointer(&cStrings[0])),
		uintptr(len(inputs)),
		uintptr(unsafe.Pointer(&outputs[0])),
		uintptr(len(outputs)),
	)
	if code != 0 {
		return nil, errorFromProc(procErr)
	}
	results := make([]int, len(outputs))
	for i, value := range outputs {
		results[i] = int(value)
	}
	return results, nil
}

func (lib *nativeLibrary) predictTextLabels(handle uintptr, inputs []string) ([]string, error) {
	cStrings, release, err := makeCStringArray(inputs)
	if err != nil {
		return nil, err
	}
	defer release()
	indices := make([]uintptr, len(inputs))
	code, _, procErr := lib.predictTextClassProc.Call(
		handle,
		uintptr(unsafe.Pointer(&cStrings[0])),
		uintptr(len(inputs)),
		uintptr(unsafe.Pointer(&indices[0])),
		uintptr(len(indices)),
	)
	if code != 0 {
		return nil, errorFromProc(procErr)
	}

	results := make([]string, len(indices))
	for i, index := range indices {
		buf := make([]byte, 256)
		code, _, err := lib.classLabelCopyProc.Call(handle, index, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
		if code != 0 {
			return nil, errorFromProc(err)
		}
		n := 0
		for n < len(buf) && buf[n] != 0 {
			n++
		}
		results[i] = string(buf[:n])
	}
	return results, nil
}

func (lib *nativeLibrary) save(handle uintptr, path string) error {
	ptr, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}
	code, _, procErr := lib.saveProc.Call(handle, uintptr(unsafe.Pointer(ptr)))
	if code != 0 {
		return errorFromProc(procErr)
	}
	return nil
}

func (lib *nativeLibrary) load(path string) (uintptr, error) {
	ptr, err := syscall.BytePtrFromString(path)
	if err != nil {
		return 0, err
	}
	var handle uintptr
	code, _, procErr := lib.loadProc.Call(uintptr(unsafe.Pointer(ptr)), uintptr(unsafe.Pointer(&handle)))
	if code != 0 {
		return 0, errorFromProc(procErr)
	}
	return handle, nil
}

func (lib *nativeLibrary) outputType(handle uintptr) (OutputType, error) {
	code, _, err := lib.outputTypeProc.Call(handle)
	if code > 2 {
		return OutputType(0), errorFromProc(err)
	}
	return OutputType(code), nil
}

func flatPtr(values []float64) uintptr {
	if len(values) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&values[0]))
}

func makeCStringArray(values []string) ([]*byte, func(), error) {
	ptrs := make([]*byte, len(values))
	buffers := make([][]byte, len(values))
	for i, value := range values {
		buffers[i] = append([]byte(value), 0)
		ptrs[i] = &buffers[i][0]
	}
	release := func() {
		runtime.KeepAlive(buffers)
	}
	return ptrs, release, nil
}

func errorFromProc(err error) error {
	if err == syscall.Errno(0) || err == nil {
		return errors.New("native call failed")
	}
	return err
}
