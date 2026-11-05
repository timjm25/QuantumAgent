// DEMO circuit: 3-qubit GHZ state. Expect ~50% 000 and ~50% 111.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
