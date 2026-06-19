package ai.simplennet;

/**
 * Output type contract for future Java bindings.
 *
 * <p>The Java runtime wrapper will bind to the stable libsimplennet C ABI.
 */
public enum OutputType {
    INT,
    FLOAT,
    CATEGORY;

    static OutputType fromOrdinal(int ordinal) {
        return values()[ordinal];
    }
}
