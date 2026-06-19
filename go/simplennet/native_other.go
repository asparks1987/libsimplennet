//go:build !windows

package simplennet

import "errors"

type nativeLibrary struct{}

func defaultLibrary() (*nativeLibrary, error) {
	return nil, errors.New("native Go bindings are currently implemented for Windows DLLs")
}
