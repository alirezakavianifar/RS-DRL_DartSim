
%%
%%  This is file `IUST-Thesis.cls',
%%  It is a modification  of Tabriz_thesis of Vahid Damanafshan,
%%	by: Mahmood Amintoosi, m.amintoosi@gmail.com
%%  http://profsite.sttu.ac.ir/mamintoosi
%%	Ver 0.1, Jan 2012

%% Original Copyright

%%  The original file was xepersian-thesis.cls by Vafa Khalighi
%%   __________________________________
%%   Copyright © 2011-2012 Vahid Damanafshan
%%  
%%   http://damanafshan.tk      
%%   http://forum.parsilatex.com
%% 
%% 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TMU-Thesis}
              [2012/01/21 v0.5
 Persian thesis document class for Tarbiat Modares University]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass[a4paper,12pt]{book}
\def\university#1{\gdef\@university{#1}}
\def\department#1{\gdef\@department{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\credits#1{\gdef\@credits{#1}}
\def\thesisdate#1{\gdef\@thesisdate{#1}}
\newcommand{\firstsupervisor}[1]{\def\@firstsupervisor{#1}}
\newcommand{\secondsupervisor}[1]{\def\@secondsupervisor{#1}}
\newcommand{\firstadvisor}[1]{\def\@firstadvisor{#1}}
\newcommand{\secondadvisor}[1]{\def\@secondadvisor{#1}}
\def\name#1{\gdef\@name{#1}}
\def\surname#1{\gdef\@surname{#1}}
\def\keywords#1{\gdef\@keywords{#1}}
\long\def\faabstract#1{\long\gdef\@faabstract{#1}}
\def\faculty#1{\gdef\@faculty{#1}}
\def\subject#1{\gdef\@subject{#1}}
\def\field#1{\gdef\@field{#1}}
\def\department#1{\gdef\@department{#1}}
\def\projectLabel#1{\gdef\@projectLabel{#1}}
\def\studentID#1{\gdef\@studentID{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\latintitle#1{\gdef\@latintitle{#1}}
\def\latinauthor#1{\gdef\@latinauthor{#1}}
\def\latindegree#1{\gdef\@latindegree{#1}}
\def\latinfaculty#1{\gdef\@latinfaculty{#1}}
\def\latinthesisdate#1{\gdef\@latinthesisdate{#1}}
\def\latinsubject#1{\gdef\@latinsubject{#1}}
\def\latinfield#1{\gdef\@latinfield{#1}}
\def\firstlatinsupervisor#1{\gdef\@firstlatinsupervisor{#1}}
\def\secondlatinsupervisor#1{\gdef\@secondlatinsupervisor{#1}}
\def\firstlatinadvisor#1{\gdef\@firstlatinadvisor{#1}}
\def\secondlatinadvisor#1{\gdef\@secondlatinadvisor{#1}}
\def\latinname#1{\gdef\@latinname{#1}}
\def\latinsurname#1{\gdef\@latinsurname{#1}}
\long\def\enabstract#1{\long\gdef\@enabstract{#1}}
\def\latinkeywords#1{\gdef\@latinkeywords{#1}}
\def\latinuniversity#1{\gdef\@latinuniversity{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@bscthesis
\@bscthesisfalse
\DeclareOption{bsc}{\@bscthesistrue}
\newif\if@mscthesis
\@mscthesisfalse
\DeclareOption{msc}{\@mscthesistrue}
\newif\if@phdthesis
\@phdthesisfalse
\DeclareOption{phd}{\@phdthesistrue}
\ProcessOptions

\ifx\@projectLabel\undefined%
{
\if@mscthesis
{\projectLabel{پایان‌نامه}}
\else
 \if@bscthesis
 {\projectLabel{پروژه }}
 \else
{\projectLabel{رساله}}
\fi\fi
}\fi

\ifx\@degree\undefined%
{
\if@mscthesis
\degree{کارشناسی ارشد}
\else
\if@bscthesis
\degree{کارشناسی}
\else
\degree{دکتری}
\fi\fi
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\firstPage}{\begin{titlepage}
%\topmargin=-30pt
\vspace{-10mm}
\centerline{{\includegraphics[width=5cm]{Images/logo}}}
\par
\vspace{-5mm}
{\large\bfseries دانشکده \@faculty}
\vspace{-5mm}

\if@bscthesis 
	{\large\bfseries{پروژه کارشناسی \@subject}
	\ifx\@field\undefined%
		\relax
	\else
		گرایش \@field
	\fi
	}
\else
	{
		\vskip .5cm
		 {\large\bfseries \@projectLabel\ \ 
		 	\@degree{} 
		 }
		 {\large\bfseries{در رشته \@subject}}
		 \vskip .2cm
		 \centerline{\large \bfseries%
		 	گرایش \@field}
	}
\fi	
\vskip 1.5cm
{\huge\titlefont \@title}\par
\vskip 5mm

\vskip 1cm
\par
{\Large\bfseries{\large\bfseries   \@name\ \ \@surname}}
\par
\vskip 15mm
\large
{
\ifx\@firstsupervisor\undefined%
	\ifx\@secondsupervisor\undefined%
	\else
	\fi
\else
\ifx\@secondsupervisor\undefined%
استاد راهنما
{\Large\bfseries\par
\@firstsupervisor}
\else
\large
اساتید راهنما
\par
{\Large\bfseries \@firstsupervisor{} و \@secondsupervisor}
\fi\fi}
\par\large
{\ifx\@firstadvisor\undefined%
\ifx\@secondadvisor\undefined%
\else\fi
\else
\ifx\@secondadvisor\undefined%
استاد مشاور
{\Large\bfseries\par
\@firstadvisor}
\else

\large
استادان مشاور
\par
{\Large\bfseries\@firstadvisor{} و \@secondadvisor}
\fi\fi}

\par
\vskip 6mm
%\vfill
{\large \@thesisdate}
%\vfill
\end{titlepage}%
} % end of firstPage

\newcommand{\besmPage}{
%% صفحه سفید
\newpage
%\pagestyle{empty}\noindent\\ 
%% بسم الله الرحمن الرحیم
\newpage
\thispagestyle{empty}
%\hfill
\begin{center}
\includegraphics[width=10cm]{Images/besm}
\end{center}
}

%% تاییدیه هیات داوران
\newcommand{\davaranPage}{
\newpage
\setcounter{page}{1}
%\thispagestyle{empty}

\begin{center}
{\Large \titlefont
 تأییدیه‌ی هیأت داوران جلسه‌ی دفاع از \@projectLabel
 }
\end{center}
\vspace{.5cm}


خانم/آقای \@name{} \@surname {}  \@projectLabel   {}  \@credits {} {}واحدی خود را با عنوان  \@title {} در تاریخ  \@thesisdate ارائه کردند.
اعضای هیات داوران نسخه نهایی این  \@projectLabel{} را از نظر فرم و محتوا تایید کرده است و پذیرش آنرا برای تکمیل درجه \@degree {} پیشنهاد می‌کنند.
%\vspace{.5cm}


}

% تأییدیه‌ی صحت و اصالت نتایج
\newcommand{\esalatPage}{
\newpage
%\thispagestyle{empty}
\begin{center}
{\Large \titlefont
آیین‌نامه حق مالکیت مادی و معنوی در مورد نتایج پژوهشهای علمی‌دانشگاه تربیت مدرس \\
 }
\vspace{.5cm}
 
\vspace{.5cm}
\end{center}
\doublespacing

مقدمه: با عنایت به سیاست‌های پژوهشی و فناوری دانشگاه در راستای تحقق عدالت و کرامت انسانها که لازمه شکوفایی علمی‌و فنی است و رعایت حقوق مادی و معنوی دانشگاه و پژوهشگران، لازم است اعضای هیأت علمی، دانشجویان، دانش‌آموختگان و دیگر همکاران طرح، در مورد نتایج پژوهشهای علمی‌که تحت عناوین پایان‌نامه‌، رساله و طرحهای تحقیقاتی با هماهنگی دانشگاه انجام شده است، موارد زیر را رعایت نمایند:

ماده 1- حق نشر و تکثیر پایان‌نامه/رساله و درآمدهای حاصل از آنها متعلق به دانشگاه می‌باشد ولی حقوق معنوی پدید آورندگان محفوظ خواهد بود.

ماده 2- انتشار مقاله یا مقالات مستخرج از پایان‌نامه/ رساله به صورت چاپ در نشریات علمی‌و یا ارائه  در مجامع علمی ‌باید به نام دانشگاه بوده و با تایید استاد راهنمای اصلی، یکی از اساتید راهنما، مشاور و یا دانشجو مسئول مکاتبات مقاله باشد. ولی مسئولیت علمی‌مقاله مستخرج از پایان نامه و رساله به عهده اساتید راهنما و دانشجو می‌باشد.
تبصره: در مقالاتی که پس از دانش‌آموختگی بصورت ترکیبی از اطلاعات جدید و نتایج حاصل از پایان‌نامه/ رساله نیز منتشر می‌شود نیز باید نام دانشگاه درج شود.

ماده 3- انتشار کتاب، نرم‌افزار و یا  آثار ویژه (اثری هنری مانند فیلم، عکس، نقاشی و نمایشنامه) حاصل از نتایج پایان‌نامه/ رساله و تمامی‌طرحهای تحقیقاتی کلیه واحدهای دانشگاه اعم از دانشکده ها، مراکز تحقیقاتی، پژوهشکده ها، پارک علم و فناوری و دیگر واحدها باید با مجوز کتبی صادره از معاونت پژوهشی دانشگاه و براساس آئین نامه های مصوب انجام شود.

ماده 4- ثبت اختراع و تدوین دانش فنی و یا ارائه یافته ها در جشنواره‌های ملی، منطقه‌ای و بین‌المللی که حاصل نتایج مستخرج از پایان‌نامه/ رساله و تمامی‌طرح‌های تحقیقاتی دانشگاه باید با هماهنگی استاد راهنما یا مجری طرح از طریق معاونت پژوهشی دانشگاه انجام گیرد.

ماده 5- این آیین‌نامه در 5 ماده و یک تبصره در تاریخ1/4/87 در شورای پژوهشی و در تاریخ 23/4/87 در هیأت رئیسه دانشگاه به تایید رسید و در جلسه مورخ 15/7/87  شورای دانشگاه به تصویب رسیده و از تاریخ تصویب در شورای دانشگاه لازم‌الاجرا است.

اینجانب 
\@name \ \@surname \ 
به شماره دانشجویی 
\@studentID {}
دانشجوی رشته 
\@subject {}
مقطع تحصیلی 
\@degree {}
دانشکده 
\@faculty{}
 متعهد می‌شوم کلِیه نکات مندرج در آئین نامه حق مالکیت مادی و معنوی در مورد نتایج پژوهش های  علمی‌دانشگاه تربیت مدرس را در انتشار یافته های علمی‌مستخرج از پایان‌نامه/رساله تحصیلی خود رعایت نمایم. در صورت تخلف از مفاد آئین نامه فوق الاشعار به دانشگاه وکالت و نمایندگی می‌دهم که از طرف اینجانب نسبت به لغو امتیاز اختراع بنام بنده و یا هر گونه امتیاز دیگر و تغییر آن به نام دانشگاه اقدام نماید. ضمناً نسبت به جبران فوری ضرر و زیان حاصله بر اساس برآورد دانشگاه اقدام خواهم نمود و بدینوسیله حق هر گونه اعتراض را از خود سلب نمودم»

\vspace{.5cm}
\begin{flushleft}
\begin{tabular}{lr}

 تاریخ و امضا: & \\
\end{tabular}
\end{flushleft}
}

% مجوز بهره‌برداری از پایان‌نامه
\newcommand{\mojavezPage}
{
\newpage
%\thispagestyle{empty}
\begin{center}
{\Large \titlefont
آیین نامه چاپ پایان‌نامه (رساله‌)های دانشجویان دانشگاه تربیت مدرس‌ \\
 }
\vspace{.5cm}
\end{center}
\doublespacing 

نظر به اینکه چاپ و انتشار پایان نامه (رساله‌)های تحصیلی دانشجویان دانشگاه تربیت مدرس‌، مبین ‌بخشی از فعالیتهای علمی‌- پژوهشی دانشگاه است بنابراین به منظور آگاهی و رعایت حقوق دانشگاه‌،دانش آموختگان این دانشگاه نسبت به رعایت موارد ذیل متعهد می‌شوند:
ماده 1: در صورت اقدام به چاپ پایان نامه (رساله‌)ی خود، مراتب را قبلاً به‌طور کتبی به «دفتر نشر آثارعلمی‌» دانشگاه اطلاع دهد.
ماده 2: در صفحه سوم کتاب (پس از برگ شناسنامه‌) عبارت ذیل را چاپ کند:
«کتاب حاضر، حاصل پایان نامه کارشناسی ارشد/ رساله دکتری نگارنده در رشته‌ \@subject {} است‌ که در \@thesisdate	{}  در دانشکده‌\@faculty {} دانشگاه تربیت مدرس به راهنمایی سرکار خانم‌/جناب ‌آقای \@firstsupervisor، مشاوره سرکار خانم‌/جناب آقای  \@firstadvisor  {} از آن دفاع شده است‌.»
ماده 3: به منظور جبران بخشی از هزینه‌های انتشارات دانشگاه‌، تعداد یک درصد شمارگان کتاب (در هر نوبت‌چاپ‌) را به «دفتر نشر آثارعلمی‌» دانشگاه اهدا کند. دانشگاه می‌تواند مازاد نیاز خود را به نفع مرکز نشر درمعرض فروش قرار دهد.
ماده 4: در صورت عدم رعایت ماده 3، 50\% بهای شمارگان چاپ شده را به‌عنوان خسارت به دانشگاه تربیت‌مدرس‌، تأدیه کند.
ماده 5: دانشجو تعهد و قبول می‌کند در صورت خودداری از پرداخت بهای خسارت‌، دانشگاه می‌تواند خسارت مذکور را از طریق مراجع قضایی مطالبه و وصول کند؛ به علاوه به دانشگاه حق می‌دهد به منظور استیفای حقوق خود، از طریق دادگاه‌، معادل وجه مذکور در ماده 4 را از محل توقیف کتابهای عرضه شده ‌نگارنده برای فروش‌، تامین نماید.
ماده 6: اینجانب
\@name \ \@surname \ {} 
دانشجوی رشته
\@subject{} 
‌مقطع‌		
\@degree{} 
تعهد فوق وضمانت اجرایی آن را قبول کرده‌، به آن ملتزم می‌شوم‌.

\vspace{.5cm}
\begin{flushleft}
\begin{tabular}{l p{.25\textwidth}}
% چون & در ماکروی دخل جدول مشکل ایجاد می‌کرد مجبور شدم دو تیکه بنویسم
\ifx\@firstsupervisor\undefined%
	\relax
\else
	\ifx\@secondsupervisor\undefined%
		استاد راهنما:
	\else
		اساتید راهنما: 
	\fi
\fi
&
\ifx\@firstsupervisor\undefined%
	\relax
\else
	\ifx\@secondsupervisor\undefined%
		\@firstsupervisor
	\else
		\@firstsupervisor{} \newline \@secondsupervisor
	\fi
\fi \\
  تاریخ: & \\
	 امضا: &\\
\end{tabular}
\end{flushleft}
}

% \def\frame{\newpage
% \thispagestyle{empty}
% \begin{framed}\thispagestyle{empty}}
% \def\endframe{\thispagestyle{empty}\end{framed}
% \thispagestyle{empty}
% \newpage
% }
\newcommand\abstractPage{\newpage
\thispagestyle{empty}
\vskip 15mm
{\noindent\bfseries چکیده}
\vspace{5mm}
\par
\noindent
\@faabstract
\vspace{8mm}
\par
\noindent
\textbf{واژگان کلیدی:} 
\@keywords
\newpage
}

% \def\latinabstract{\newpage
% \thispagestyle{empty}
% \vskip 15mm
% \begin{center}%{\Large\bfseries \@latintitle \\[5mm]}
% {\bfseries Abstract}
% \end{center}
% \vspace{5mm}}

% \def\endlatinabstract{\newpage}

\def\acknowledgementpage{\newpage
\thispagestyle{empty}
\centerline{\Large \bfseries قدردانی}
\vspace{1cm}
\par\noindent}
\def\endacknowledgementpage{\newpage}

\def\titlepage{\newpage\centering
  \thispagestyle{empty}
  \parindent 0pt \parskip 10pt plus 1fil minus 1fil
  \def\baselinestretch{1}\@normalsize\vbox to \vsize\bgroup\vbox to 9in\bgroup}
\def\endtitlepage{\par\kern 0pt\egroup\vss\egroup\newpage}

\def\signature{\vspace{1cm}
\begin{flushleft}
{%\nastaliq \scriptsize 
\@name\ \@surname \\
\@thesisdate}
\end{flushleft}}

% \def\abstract{
% \newpage
% \subsection*{چکیده}\small
% \@normalsize
% \@fa-abstract
% }
% \def\endabstract{\par}

%\pagenumbering{harfi}
%\let\ol@chapter\@chapter
%\def\@chapter{%
%  \ifnum\c@chapter=0 \pagenumbering{arabic}\setcounter{page}{1}\fi
%  \ol@chapter}

%\font\latintitlefont=cmssbx10 scaled 2300 %cmssbx10 scaled 2300
\font\supervisorfont=cmbxti10

\newcommand{\latinfirstPage}{%
\begin{latin}
\newpage\clearpage
\thispagestyle{empty}
\vskip 15mm
{\noindent\bfseries Abstract:}
\vspace{5mm}
\par
\noindent
\@enabstract
\vspace{8mm}
\par
\noindent
\textbf{Keywords:} 
\@latinkeywords

\begin{titlepage}
\vspace{-10mm}
{\includegraphics[height=4.5cm]{Images/logo-en}}
\centerline{\bf \@latinuniversity} 
\large\bfseries
\@latinfaculty\space 
\vskip 1cm
{%\latintitlefont 
\Huge
\@latintitle}
%\baselineskip=.750cm \par
\vskip 15mm
{
\large
\if@bscthesis 
    Bachelor of Science Thesis
	in  \@latinsubject 
\else
	A Thesis Submitted in Partial Fulfillment of the Requirement for the Degree of 
	 \if@mscthesis
	Master of Science
	\else
	 Doctor of Philosophy
	\fi
	in  \@latinsubject 
	%\baselineskip=.750cm
	%\addtolength{\baselineskip}{-5mm}
\fi
}
\vskip 15mm
{\bfseries\latinby}\par
{\large %\supervisorfont
\rm\@latinname \; \@latinsurname}\par

\vskip 15mm

\large
{\ifx\@firstlatinsupervisor\undefined%
\ifx\@secondlatinsupervisor\undefined%
\else\fi
\else
\ifx\@secondlatinsupervisor\undefined%
Supervisor:
{\Large\bfseries\par
\@firstlatinsupervisor}
\else
\large
Supervisors:
\par
{\Large\bfseries \@firstlatinsupervisor{} and \@secondlatinsupervisor}
\fi\fi}
\par
\large
{\ifx\@firstlatinadvisor\undefined%
\ifx\@secondlatinadvisor\undefined%
\else\fi
\else
\ifx\@secondlatinadvisor\undefined%
Advisor:
{\Large\bfseries\par
\@firstlatinadvisor}
\else

\large
Advisors:
\par
{\Large\bfseries\@firstlatinadvisor{} and \@secondlatinadvisor}
\fi\fi}
\vskip 5mm

\par
\vskip 1cm

{\@latinthesisdate}
\par
\vskip 1cm

\par

\vfill
%
\end{titlepage}
\end{latin}
}
\fi

\def\onvan{عنوان}
\def\writtenfor{برای دریافت درجه }
\def\by{پژوهشگر}
\def\latinby{By:}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

%% M.Amintoosi
%% 
%% Copyright © 2011-2012      by Vahid Damanafshan     
%% 
%% Distributable under the LaTeX Project Public License,
%% version 1.3c or higher (your choice). The latest version of
%% this license is at: http://www.latex-project.org/lppl.txt
%% 
%% This work is "maintained" (as per LPPL maintenance status)
%%  by Vahid Damanafshan.
%% 
%% 
%% 
%% 
%% Special thanks to Vafa Khalighi for his invaluable hints.
%%
%% End of file `IUST-Thesis.cls'.
