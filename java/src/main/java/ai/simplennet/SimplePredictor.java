package ai.simplennet;

import java.nio.file.Path;
import java.util.List;

/**
 * Java bindings for the libsimplennet native core.
 */
public final class SimplePredictor implements AutoCloseable {
    static {
        System.loadLibrary("simplennet_native");
    }

    private long nativeHandle;
    private final OutputType outputType;

    public SimplePredictor(OutputType outputType) {
        this(nativeCreate(outputType.ordinal()), outputType);
    }

    private SimplePredictor(long nativeHandle, OutputType outputType) {
        this.nativeHandle = nativeHandle;
        this.outputType = outputType;
    }

    public OutputType outputType() {
        return outputType;
    }

    public SimplePredictor fit(double[][] inputs, double[] targets) {
        return fit(inputs, targets, 30, 0.01);
    }

    public SimplePredictor fit(double[][] inputs, double[] targets, int epochs, double learningRate) {
        nativeFitNumeric(nativeHandle, inputs, targets, epochs, learningRate);
        return this;
    }

    public SimplePredictor fitLabels(double[][] inputs, List<String> targets) {
        return fitLabels(inputs, targets, 30, 0.01);
    }

    public SimplePredictor fitLabels(double[][] inputs, List<String> targets, int epochs, double learningRate) {
        nativeFitLabels(nativeHandle, inputs, targets.toArray(new String[0]), epochs, learningRate);
        return this;
    }

    public double[] predictNumbers(double[][] inputs) {
        return nativePredictNumbers(nativeHandle, inputs);
    }

    public int[] predictInts(double[][] inputs) {
        return nativePredictInts(nativeHandle, inputs);
    }

    public String[] predictLabels(double[][] inputs) {
        return nativePredictLabels(nativeHandle, inputs);
    }

    public void save(Path path) {
        nativeSave(nativeHandle, path.toString());
    }

    public static SimplePredictor load(Path path) {
        long handle = nativeLoad(path.toString());
        OutputType outputType = OutputType.fromOrdinal(nativeOutputType(handle));
        return new SimplePredictor(handle, outputType);
    }

    public String[] classLabels() {
        return nativeClassLabels(nativeHandle);
    }

    @Override
    public void close() {
        if (nativeHandle != 0) {
            nativeDestroy(nativeHandle);
            nativeHandle = 0;
        }
    }

    private static native long nativeCreate(int outputTypeOrdinal);
    private static native void nativeDestroy(long handle);
    private static native void nativeFitNumeric(long handle, double[][] inputs, double[] targets, int epochs, double learningRate);
    private static native void nativeFitLabels(long handle, double[][] inputs, String[] targets, int epochs, double learningRate);
    private static native double[] nativePredictNumbers(long handle, double[][] inputs);
    private static native int[] nativePredictInts(long handle, double[][] inputs);
    private static native String[] nativePredictLabels(long handle, double[][] inputs);
    private static native void nativeSave(long handle, String path);
    private static native long nativeLoad(String path);
    private static native int nativeOutputType(long handle);
    private static native String[] nativeClassLabels(long handle);
}
