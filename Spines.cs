namespace PF1;

public class SplinesFunc : MatFuncs {

    ////////////////// GIVEN

    public static List<qmatrix<double>> MatSplineF(List<int> L1, List<double> L2) {
        int n1 = L1.Count;
        int n2 = L2.Count;

        CAssert.Assert(n1 == n2, $"ERREUR: Les listes doivent être de même longueur. {n1} != {n2}");
        CAssert.Assert(L2[0] == L2[n2 - 1], "ERREUR: Ne reflète pas les données pour une spline fermée.");

        var M = MatNulle<double>(4 * (n1 - 1), 4 * (n1 - 1));

        for (int i = 0; i < n1 - 1; i++) {
            for (int j = 0; j < n1 - 1; j++) {
                M.mat[2 * i][4 * j] = 1;
                M.mat[2 * i][4 * j + 1] = L1[i];
                M.mat[2 * i][4 * j + 2] = Math.Pow(L1[i], 2);
                M.mat[2 * i][4 * j + 3] = Math.Pow(L1[i], 3);
                M.mat[2 * i + 1][4 * j] = 1;
                M.mat[2 * i + 1][4 * j + 1] = L1[i + 1];
                M.mat[2 * i + 1][4 * j + 2] = Math.Pow(L1[i + 1], 2);
                M.mat[2 * i + 1][4 * j + 3] = Math.Pow(L1[i + 1], 3);

                i++;
            }
        }
        // Les informations sur les dérivées premières aux valeurs intermédiaires

        for (int j = 0; j < n1 - 2; j++) {
            for (int i = 0; i < n1 - 2; i++) {
                M.mat[2 * (n1 - 1) + i][4 * j] = 0;
                M.mat[2 * (n1 - 1) + i][4 * j + 1] = 1;
                M.mat[2 * (n1 - 1) + i][4 * j + 2] = 2 * L1[i + 1];
                M.mat[2 * (n1 - 1) + i][4 * j + 3] = 3 * Math.Pow(L1[i + 1], 2);
                M.mat[2 * (n1 - 1) + i][4 * j + 4] = 0;
                M.mat[2 * (n1 - 1) + i][4 * j + 5] = -1;
                M.mat[2 * (n1 - 1) + i][4 * j + 6] = -2 * L1[i + 1];
                M.mat[2 * (n1 - 1) + i][4 * j + 7] = -3 * Math.Pow(L1[i + 1], 2);
                j++;
            }
        }
        // Les informations sur les dérivées secondes aux valeurs intermédiaires
        for (int j = 0; j < n1 - 2; j++) {
            for (int i = 0; i < n1 - 2; i++) {
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 1] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 2] = 2;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 3] = 6 * L1[i + 1];
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 4] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 5] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 6] = -2;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 7] = -6 * L1[i + 1];
                j++;
            }
        }

        // Il manque deux lignes qui correspondent au fait de traiter des splines fermées.
        // *Ce sont ces deux lignes seulement qui changeront pour les splines naturelles.

        // L'avant dernière ligne
        M.mat[4 * (n1 - 1) - 2][0] = 0;
        M.mat[4 * (n1 - 1) - 2][1] = 1;
        M.mat[4 * (n1 - 1) - 2][2] = 2 * L1[0];
        M.mat[4 * (n1 - 1) - 2][3] = 3 * Math.Pow(L1[0], 2);
        M.mat[4 * (n1 - 1) - 2][4 * (n1 - 1) - 4] = 0;
        M.mat[4 * (n1 - 1) - 2][4 * (n1 - 1) - 3] = -1;
        M.mat[4 * (n1 - 1) - 2][4 * (n1 - 1) - 2] = -2 * L1[n1 - 1];
        M.mat[4 * (n1 - 1) - 2][4 * (n1 - 1) - 1] = -3 * Math.Pow(L1[n1 - 1], 2);

        // La dernière ligne
        M.mat[4 * (n1 - 1) - 1][0] = 0;
        M.mat[4 * (n1 - 1) - 1][1] = 0;
        M.mat[4 * (n1 - 1) - 1][2] = 2;
        M.mat[4 * (n1 - 1) - 1][3] = 6 * L1[0];
        M.mat[4 * (n1 - 1) - 1][4 * (n1 - 1) - 4] = 0;
        M.mat[4 * (n1 - 1) - 1][4 * (n1 - 1) - 3] = 0;
        M.mat[4 * (n1 - 1) - 1][4 * (n1 - 1) - 2] = -2;
        M.mat[4 * (n1 - 1) - 1][4 * (n1 - 1) - 1] = -6 * L1[n1 - 1];

        // Création de la matrice des coefficients qui ne prend en compte de la liste L2.
        var N = MatNulle<double>(4 * (n2 - 1), 1);

        N.mat[0][0] = L2[0];
        N.mat[2 * (n2 - 1) - 1][0] = L2[n2 - 1];

        for (int i = 0; i < n2 - 2; i++) {
            N.mat[2 * i + 1][0] = L2[i + 1];
            N.mat[2 * i + 2][0] = L2[i + 1];
        }

        List<qmatrix<double>> Rep = [M, N];
        return Rep;
    }

    // WROTE MYSELF

    public static List<qmatrix<double>> MatSplineN(List<int> L1, List<double> L2) {
        int n1 = L1.Count;
        int n2 = L2.Count;

        if (n1 != n2) throw new Exception("ERREUR: Les listes doivent être de même longueur.");
        if (L2[0] != L2[n2 - 1]) throw new Exception("ERREUR: Ne reflète pas les données pour une spline fermée.");

        var M = MatNulle<double>(4 * (n1 - 1), 4 * (n1 - 1));

        for (int i = 0; i < n1 - 1; i++) {
            for (int j = 0; j < n1 - 1; j++) {
                M.mat[2 * i][4 * j] = 1;
                M.mat[2 * i][4 * j + 1] = L1[i];
                M.mat[2 * i][4 * j + 2] = Math.Pow(L1[i], 2);
                M.mat[2 * i][4 * j + 3] = Math.Pow(L1[i], 3);
                M.mat[2 * i + 1][4 * j] = 1;
                M.mat[2 * i + 1][4 * j + 1] = L1[i + 1];
                M.mat[2 * i + 1][4 * j + 2] = Math.Pow(L1[i + 1], 2);
                M.mat[2 * i + 1][4 * j + 3] = Math.Pow(L1[i + 1], 3);

                i++;
            }
        }
        for (int j = 0; j < n1 - 2; j++) {
            for (int i = 0; i < n1 - 2; i++) {
                M.mat[2 * (n1 - 1) + i][4 * j] = 0;
                M.mat[2 * (n1 - 1) + i][4 * j + 1] = 1;
                M.mat[2 * (n1 - 1) + i][4 * j + 2] = 2 * L1[i + 1];
                M.mat[2 * (n1 - 1) + i][4 * j + 3] = 3 * Math.Pow(L1[i + 1], 2);
                M.mat[2 * (n1 - 1) + i][4 * j + 4] = 0;
                M.mat[2 * (n1 - 1) + i][4 * j + 5] = -1;
                M.mat[2 * (n1 - 1) + i][4 * j + 6] = -2 * L1[i + 1];
                M.mat[2 * (n1 - 1) + i][4 * j + 7] = -3 * Math.Pow(L1[i + 1], 2);
                j++;
            }
        }
        for (int j = 0; j < n1 - 2; j++) {
            for (int i = 0; i < n1 - 2; i++) {
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 1] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 2] = 2;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 3] = 6 * L1[i + 1];
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 4] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 5] = 0;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 6] = -2;
                M.mat[2 * (n1 - 1) + (n1 - 2) + i][4 * j + 7] = -6 * L1[i + 1];
                j++;
            }
        }

        //changed
        M.mat[4 * (n1 - 1) - 2][2] = 2;
        M.mat[4 * (n1 - 1) - 2][3] = 6 * L1[0];
        M.mat[4 * (n1 - 1) - 1][4 * (n1 - 1) - 2] = 2;
        M.mat[4 * (n1 - 1) - 1][4 * (n1 - 1) - 1] = 6 * L1[n1 - 1];

        var N = MatNulle<double>(4 * (n2 - 1), 1);
        N.mat[0][0] = L2[0];
        N.mat[2 * (n2 - 1) - 1][0] = L2[n2 - 1];
        for (int i = 0; i < n2 - 2; i++) {
            N.mat[2 * i + 1][0] = L2[i + 1];
            N.mat[2 * i + 2][0] = L2[i + 1];
        }
        List<qmatrix<double>> Rep = [M, N];
        return Rep;
    }
    public static qmatrix<double> SplineF(List<int> L1, List<double> L2) {
        List<qmatrix<double>> Matrices = MatSplineF(L1, L2);
        qmatrix<double> M = Matrices[0];  // Matrix of the system
        qmatrix<double> N = Matrices[1];  // Right-hand side (L2)

        // solve M * C = N for C 
        var Minv = MatInvW(M);
        var C = MatProduit(Minv, N);

        return C;
    }
    public static qmatrix<double> SplineN(List<int> L1, List<double> L2) {
        List<qmatrix<double>> Matrices = MatSplineN(L1, L2);
        qmatrix<double> M = Matrices[0];  // matrix of the system
        qmatrix<double> N = Matrices[1];  // right-hand side (L2)
        // solve M * C = N for C 
        var Minv = MatInvLU(M);
        var C = MatProduit(Minv, N);
        return C;
    }
}