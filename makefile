## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-cli python-doc python-doc-deploy
INFO_TARGETS +=		appinfo


## Project
#
ENTRY =			./calsum


## Includes
#
include ./zenbuild/main.mk


## Targets
#
.PHONY:			appinfo
appinfo:
			@echo "app-resources-dir: $(RESOURCES_DIR)"


# align admissions
.PHONY:			align
align:
			$(ENTRY) clear
			nohup $(ENTRY) align > align.log 2>&1 &

# print out a UI Health admission
.PHONY:			showuic
showuic:
			$(ENTRY) show 22151226

.PHONY:			pulluic
pulluic:
			rm -fr config/csv/uic
			mkdir -p config/csv
			hostcon pull -n acer /view/nlp/med/sum-calsum/config/csv/ -l config/csv/uic
			hostcon pull -n acer /view/nlp/med/sum-calsum/image/match-confusion-matrix.svg -l image/uic-match-confusion-matrix.svg
			hostcon pull -n acer /view/nlp/med/sum-calsum/image/note-sec-contingency.svg -l image/uic-note-sec-contingency.svg

# stop the current running process
.PHONY:			stop
stop:
			@ps -eaf | grep -i python | grep calsum | grep -v grep | \
				awk '{print $$2}' | xargs kill || true
			@ps -eaf | grep -i python | grep multiprocessing | grep -v grep | \
				awk '{print $$2}' | xargs kill || true
