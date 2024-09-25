using PF1;

static void WRT<T>(T t) => Console.WriteLine(t); // wrapper to make Console.Writeline shorter to use

var A = new Matrix<double>([
    [ -35,   0, -19,  99, -91, -35,   0],
    [   0,   0, -50,   0,   0,  80,  12],
    [   0, -88,   0,  -3,   0, -82,   0],
    [   0,   0,   0,   0,   0,  91,  22],
    [ -46,  18, -97,  95, -26,   0,   9],
    [   0,  86,   0, -61,  22,   0,   0],
    [   0,   0, -15, -78,   0,  82, -25]]);

var B = new Matrix<double>([
    [  0,   1,   9],
    [  0,   0,   0],
    [ -1,   2,   0],
    [ 27,  75,   2],
    [ 54,   0, -29],
    [  0, -45,   0],
    [  0,   1,   0]]);


WRT("AB");
WRT(M.MatProduit(A, B));
WRT("");

WRT("A^4");
WRT(M.MatPow(A, 4));
WRT("");

WRT("|A|");
WRT(M.MatDetW(A));
WRT("");


WRT("A^-1");
WRT(M.MatInvW(A));
WRT("");


// Matrix<T>        MatNulle<T> (int rows, int cols)
// Matrix<double>   MatRand     (int rows, int cols)
// Matrix<double>   MatId       (int x)
// Matrix<double>   MatT        (Matrix<double> A)
// Matrix<double>   MatSomme    (Matrix<double> A, Matrix<double> B)
// Matrix<double>   MatMultk    (Matrix<double> A, double k)
// Matrix<double>   MatProduit  (Matrix<double> A, Matrix<double> B)
// Matrix<double>   MatPow      (Matrix<double> A, double k)
// double           MatDetW     (Matrix<double> M)
// Matrix<double>   MatSousMat  (Matrix<double> matrix, int rowToRemove, int colToRemove)
// Matrix<double>   MatInvW     (Matrix<double> A)