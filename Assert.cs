#pragma warning disable CS8981 // The type name only contains lower-cased ascii characters. Such names may become reserved for the language.

using System.Diagnostics;

namespace PF1 {
    public class AssertException(string message) : Exception(message) { }
    public static class CAssert {
        public static void Assert(bool condition, string message) {
            if (!condition) {
                // Console.Error.WriteLine(message);
                throw new AssertException(message + "\n\n");
            }
        }
    }
}