
package:
	rm -rf /tmp/buildS
	mkdir /tmp/buildS
	cp LICENSE README.md  /tmp/buildS
	cp setup.py /tmp/buildS/setup.py
	cp -r GenericSolver/python /tmp/buildS/genericSolver
	cd /tmp/buildS && \
	python setup.py check &&\
	python setup.py sdist

