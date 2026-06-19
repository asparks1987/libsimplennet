//go:build windows

package simplennet

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSimplePredictorWindows(t *testing.T) {
	dll := os.Getenv("SIMPLENET_NATIVE_LIBRARY")
	if dll == "" {
		dll = filepath.Join("..", "..", "build-native", "Release", "simplennet_native.dll")
	}
	if _, err := os.Stat(dll); err != nil {
		t.Skipf("native library not available: %v", err)
	}
	if err := os.Setenv("SIMPLENET_NATIVE_LIBRARY", dll); err != nil {
		t.Fatal(err)
	}

	predictor, err := NewSimplePredictor(OutputInt)
	if err != nil {
		t.Fatal(err)
	}
	defer predictor.Close()

	if err := predictor.Fit([][]float64{{0}, {1}, {2}, {3}}, []float64{0, 2, 4, 6}); err != nil {
		t.Fatal(err)
	}

	values, err := predictor.PredictInts([][]float64{{4}})
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 1 {
		t.Fatalf("expected 1 value, got %d", len(values))
	}
}
