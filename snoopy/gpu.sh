ssh ec22201@login.eecs.qmul.ac.uk
ssh ec22201@prospero
server: grover
Rethoethibie2teeP6ie
genv

ssh -NL 9000:<compute server>:9000 <username>@grover.eecs.qmul.ac.uk
ssh -NL 9000:prospero:9000 ec22201@grover.eecs.qmul.ac.uk
ssh -NL 9000:grover:9000 ec22201@grover.eecs.qmul.ac.uk


hostname: prospero
username: ec22201
Gpu: IP: 138.37.37.79


ssh -NL 9000:prospero.eecs.qmul.ac.uk:9000 ec22201@grover.eecs.qmul.ac.uk

ssh -NL 9000:prospero:9000 ec22201@prospero.eecs.qmul.ac.uk


curl -L https://code.visualstudio.com/download/linux/latest/installer -o vscode.deb