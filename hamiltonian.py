class HamiltonianPlugin:
    """
    Plugin for generating the active-space Hamiltonian for Ammonia (NH3).
    """
    MIN_TERMS = 400
    PAD_SYNTHETIC = True
    SYN_COEFF_SCALE = 1e-8

    def __init__(self):
        # Default to a reasonable NH3 geometry (approximate, Angstroms)
        # If you want H2 instead, replace with ["H 0 0 0", "H 0 0 0.74"] or multiline string.
        self.geom = [
            "N 0.000000 0.000000 0.000000",
            "H 0.000000 0.937700 -0.381600",
            "H 0.812100 -0.468800 -0.381600",
            "H -0.812100 -0.468800 -0.381600"
        ]
        #self.geom = ["H 0 0 0", "H 0 0 0.74"]
        self._hamiltonian = None
        self._problem_active = None
        self._mapper = None
        self.is_fallback = False  # Tracks whether synthetic Hamiltonian was used

    def _normalize_geometry(self, geom):
        """Return geometry in a PySCFDriver-acceptable form (string or list[str])."""
        if isinstance(geom, str):
            return geom
        if isinstance(geom, (list, tuple)):
            if all(isinstance(x, str) for x in geom):
                return "\n".join(geom)
        raise ValueError("Geometry must be str or sequence of str specifications.")

    def get_hamiltonian(self):
        """
        Builds and returns a dictionary containing the NH3 active-space Hamiltonian
        and other metadata. The result is cached to avoid re-computation.
        """
        if self._hamiltonian is not None:
            return {
                "problem_active": self._problem_active,
                "mapper": self._mapper,
                "hamiltonian_active": self._hamiltonian,
                "num_qubits": self._hamiltonian.num_qubits,
                "basis": "sto3g",
                "geometry": self.geom
            }

        # Optional hard bypass: skip PySCFDriver entirely and build via pyscf + from_pyscf path
        if os.environ.get('FORCE_FROM_PYSCF', '0') == '1':
            try:
                # Attempt modern from_pyscf helper if available, else degrade gracefully to driver logic
                from pyscf import gto, scf  # type: ignore
                geom_str = self._normalize_geometry(self.geom)
                mol = gto.M(atom=geom_str, basis='sto-3g', unit='Angstrom')
                mf = scf.RHF(mol).run()
                ham_active = None
                full_map_success = False
                try:
                    # Newer qiskit-nature versions expose a formats.pyscf import; 0.7.2 may not.
                    from qiskit_nature.second_q.formats.pyscf import from_pyscf  # type: ignore
                    result_alt = from_pyscf(mf, include_dipole=False)
                    problem_full_alt = ElectronicStructureProblem(result_alt)
                    # Attempt MO basis conversion if needed
                    try:
                        from qiskit_nature.second_q.properties import ElectronicBasis  # type: ignore
                        ei = getattr(problem_full_alt.hamiltonian, 'electronic_integrals', None)
                        if ei and hasattr(ei, 'convert_basis') and getattr(problem_full_alt.hamiltonian, 'electronic_basis', None) != ElectronicBasis.MO:
                            ei.convert_basis(ElectronicBasis.AO, ElectronicBasis.MO)
                    except Exception:
                        pass
                    transformer_alt = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
                    self._problem_active = transformer_alt.transform(problem_full_alt)
                    self._mapper = JordanWignerMapper()
                    ham2_alt = self._problem_active.second_q_ops()['ElectronicEnergy']
                    ham_active = self._mapper.map(ham2_alt)
                    if ham_active.num_qubits == 6:
                        full_map_success = True
                except Exception:
                    pass
                if full_map_success:
                    self._hamiltonian = ham_active
                    self.PAD_SYNTHETIC = False
                    self.is_fallback = False
                    print('[Info] Built Hamiltonian via FORCE_FROM_PYSCF path (full 2e terms).')
                    return {
                        "problem_active": self._problem_active,
                        "mapper": self._mapper,
                        "hamiltonian_active": self._hamiltonian,
                        "num_qubits": self._hamiltonian.num_qubits,
                        "basis": "sto3g",
                        "geometry": self.geom,
                        "fallback": False
                    }
                else:
                    print('[Warning] FORCE_FROM_PYSCF full mapping unavailable; reverting to standard driver path.')
            except Exception as force_e:
                print(f"[Warning] FORCE_FROM_PYSCF path failed early: {force_e}; continuing with standard logic.")

        try:
            if not QISKIT_NATURE_INSTALLED:
                raise ImportError("Qiskit Nature or its dependencies are not installed.")

            atom_spec = self._normalize_geometry(self.geom)
            driver = PySCFDriver(atom=atom_spec, basis='sto3g', charge=0, spin=0, unit=DistanceUnit.ANGSTROM)

            # Monkey patch missing legacy attribute if downstream code expects it
            if not hasattr(driver, 'register_length'):
                try:
                    raw_res_tmp = driver.run()
                    guess_len = getattr(raw_res_tmp, 'num_spatial_orbitals', 0) * 2
                    driver.register_length = guess_len  # type: ignore
                except Exception:
                    driver.register_length = 0  # type: ignore

            raw_res = driver.run()
            # Some versions return ElectronicStructureProblem directly; if not, wrap
            problem_full = raw_res if isinstance(raw_res, ElectronicStructureProblem) else ElectronicStructureProblem(raw_res)
            transformer = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
            try:
                self._problem_active = transformer.transform(problem_full)
            except Exception as t_e:
                # Fallback: attempt direct integral reconstruction
                raise RuntimeError(f"Active space transform failed ({t_e})")

            self._mapper = JordanWignerMapper()
            ham2 = self._problem_active.second_q_ops()['ElectronicEnergy']
            ham_active = self._mapper.map(ham2)
            if ham_active.num_qubits != 6:
                raise RuntimeError(f'Active space produced {ham_active.num_qubits} qubits, expected 6.')

        except Exception as e:
            print(f'[Warning] Ab initio build failed: {e}. Attempting direct PySCF fallback...')
            direct_pyscf_failed = False
            ham_active = None
            if 'PySCFDriver' in str(e) or 'register_length' in str(e):
                try:
                    import pyscf  # type: ignore
                    from pyscf import gto, scf  # type: ignore
                    geom_str = self._normalize_geometry(self.geom)
                    mol = gto.Mole()
                    mol.build(atom=geom_str, basis='sto-3g', unit='Angstrom')
                    mf = scf.RHF(mol)
                    e_hf = mf.kernel()

                    # First attempt: full mapped Hamiltonian via from_pyscf (retains 2e correlations)
                    full_map_success = False
                    try:
                        from qiskit_nature.second_q.formats.pyscf import from_pyscf  # type: ignore
                        result_alt = from_pyscf(mf, include_dipole=False)
                        problem_full_alt = ElectronicStructureProblem(result_alt)
                        transformer_alt = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
                        self._problem_active = transformer_alt.transform(problem_full_alt)
                        self._mapper = JordanWignerMapper()
                        ham2_alt = self._problem_active.second_q_ops()['ElectronicEnergy']
                        ham_active = self._mapper.map(ham2_alt)
                        if ham_active.num_qubits == 6:
                            full_map_success = True
                            self.PAD_SYNTHETIC = False
                            self.is_fallback = False
                            print('[Info] Recovered full active-space Hamiltonian via from_pyscf fallback (includes 2e terms).')
                        else:
                            print(f'[Info] from_pyscf produced {ham_active.num_qubits} qubits (expected 6); discarding.')
                            ham_active = None
                    except Exception as map_e:
                        print(f'[Info] from_pyscf path unavailable ({map_e}); reverting to diagonal HF model.')

                    if not full_map_success:
                        # NEW: attempt integral-based correlated Hamiltonian before diagonal simplification
                        try:
                            from qiskit_nature.second_q.hamiltonians import ElectronicEnergy  # type: ignore
                            from qiskit_nature.second_q.operators import FermionicOp  # type: ignore
                            # AO integrals
                            h1_ao = mf.get_hcore()
                            from pyscf import ao2mo  # type: ignore
                            eri_ao = ao2mo.full(mf._eri, mf.mo_coeff)  # MO two-electron (chemist)
                            # Build one- and two-body in MO basis
                            C = mf.mo_coeff
                            h1_mo = C.T @ h1_ao @ C
                            nmo = h1_mo.shape[0]
                            # Reshape two-electron integrals (chemist notation) (ij|kl)
                            eri_mo = ao2mo.restore(1, eri_ao, nmo)
                            # ElectronicEnergy helper
                            ee = ElectronicEnergy.from_raw_integrals(h1_mo, eri_mo)
                            problem_full_alt2 = ElectronicStructureProblem(ee)
                            transformer_alt2 = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3)
                            self._problem_active = transformer_alt2.transform(problem_full_alt2)
                            self._mapper = JordanWignerMapper()
                            ham2_alt2 = self._problem_active.second_q_ops()['ElectronicEnergy']
                            ham_active = self._mapper.map(ham2_alt2)
                            if ham_active.num_qubits == 6:
                                print('[Info] Built correlated Hamiltonian from raw PySCF integrals (includes 2e terms).')
                                self.PAD_SYNTHETIC = False
                                self.is_fallback = False
                            else:
                                ham_active = None
                        except Exception as int_e:
                            # Final diagonal model
                            from qiskit.quantum_info import SparsePauliOp as _SPO  # type: ignore
                            mo_energies = list(mf.mo_energy)
                            n_spatial_target = 3
                            if len(mo_energies) < n_spatial_target:
                                n_spatial_target = len(mo_energies)
                            active_eps = mo_energies[:n_spatial_target]
                            spin_eps = []
                            for eps in active_eps:
                                spin_eps.extend([float(eps), float(eps)])
                            num_qubits = len(spin_eps)
                            n_electrons = 4
                            occ_indices = list(range(min(n_electrons, len(spin_eps))))
                            sum_eps_occ = sum(spin_eps[i] for i in occ_indices)
                            const_shift = e_hf - sum_eps_occ
                            paulis = ['I'*num_qubits]
                            coeffs = [const_shift]
                            for p, eps in enumerate(spin_eps):
                                paulis.append('I'*num_qubits); coeffs.append(eps/2.0)
                                z_string = ['I']*num_qubits; z_string[p] = 'Z'
                                paulis.append(''.join(z_string)); coeffs.append(-eps/2.0)
                            ham_active = _SPO(paulis, coeffs)
                            print(f'[Info] Direct PySCF HF energy = {e_hf:.6f} Hartree (diagonal orbital-energy Hamiltonian)')
                            print('[Info] Using simplified diagonal Hamiltonian (no two-electron correlations).')
                            self.PAD_SYNTHETIC = False
                            self.is_fallback = True
                except Exception as de2:
                    direct_pyscf_failed = True
                    print(f'[Warning] Direct PySCF fallback failed: {de2}')
            if ham_active is None:
                print('Using a synthetic 6-qubit operator.')
            if not QISKIT_NATURE_INSTALLED:
                print('[Info] qiskit-nature not detected. Install with: pip install qiskit-nature pyscf')
            else:
                print('[Info] Check geometry formatting or PySCF availability.')
            if ham_active is None:
                paulis = ['IIIIII', 'ZIIIZZ', 'ZZIIZZ', 'IZZIIZ', 'IIZZZZ', 'XXYYZZ', 'YYXXZZ']
                coeffs = [-5.0, 0.12, -0.08, 0.05, -0.03, 0.01, 0.01]
                ham_active = SparsePauliOp(paulis, coeffs)
            self._problem_active = None
            self._mapper = None
            self.is_fallback = True

        terms = {str(p): complex(c) for p, c in zip(ham_active.paulis, ham_active.coeffs) if abs(complex(c)) > 1e-12}
        physical_count = len(terms)

    # (Pruned) Synthetic padding disabled

        self._print_summary(terms, physical_count)
        
        self._hamiltonian = SparsePauliOp.from_list(list(terms.items()))
        
        return {
            "problem_active": self._problem_active,
            "mapper": self._mapper,
            "hamiltonian_active": self._hamiltonian,
            "num_qubits": self._hamiltonian.num_qubits,
            "basis": "sto3g",
            "geometry": self.geom,
            "fallback": self.is_fallback
        }

    def was_fallback(self):
        """Return True if synthetic fallback Hamiltonian was used."""
        return self.is_fallback

    def info(self):
        """Return a concise diagnostic dictionary about the Hamiltonian state."""
        return {
            'fallback': self.is_fallback,
            'num_qubits': self._hamiltonian.num_qubits if self._hamiltonian else None,
            'active_problem': self._problem_active is not None,
            'mapper': self._mapper.__class__.__name__ if self._mapper else None,
            'geometry_lines': len(self.geom) if isinstance(self.geom, (list, tuple)) else 1
        }

    # (Removed) _add_synthetic_padding helper
    
    def _print_summary(self, terms, physical_count):
        total = len(terms)
        print('\n--- Hamiltonian Generation Summary ---')
        print(f'Physical (mapped) terms: {physical_count}')
        if total > physical_count:
            print(f'Synthetic padding terms added: {total - physical_count}')
        print(f'Total terms in operator: {total}')
        print('-------------------------------------\n')


