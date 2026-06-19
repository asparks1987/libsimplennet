# Java And Go

Java and Go are runtime-supported through the native build artifacts.

## Java

The Java wrapper lives in `java/src/main/java/ai/simplennet`.

```java
SimplePredictor predictor = new SimplePredictor(OutputType.INT);
predictor.fit(inputs, targets);
int[] values = predictor.predictInts(inputs);
```

Run with:

```bash
java -Djava.library.path=build-native/Release -cp java/target/classes ai.simplennet.SmokeTest
```

Java loads `simplennet_native`, the JNI bridge built when CMake finds a JDK.

## Go

The Go wrapper lives in `go/simplennet`.

```go
predictor, err := simplennet.NewSimplePredictor(simplennet.OutputInt)
if err != nil {
    panic(err)
}
defer predictor.Close()
_ = predictor.Fit(inputs, targets)
values, _ := predictor.PredictInts(inputs)
```

Set the native DLL path before running Go tests:

```powershell
$env:SIMPLENET_NATIVE_LIBRARY = "D:\path\to\build-native\Release\simplennet.dll"
go test ./...
```

## Binding Boundary

Go uses the language-neutral C ABI exported by `simplennet`. Java uses the JNI bridge, which delegates into the same C++ core. Other languages should bind to `include/simplennet/c_api.h`.
