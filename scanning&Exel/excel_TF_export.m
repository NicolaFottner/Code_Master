
if illiterate ||least_square
    filename = "excel_files/" +  "trialData." + "xlsx";
    writetable(matrix_1,filename,'WriteRowNames',true,'Sheet','Matrix_1')
    writetable(matrix_1_pd,filename,'WriteRowNames',true,'Sheet','Matrix_1pd')
    writetable(matrix_2,filename,'WriteRowNames',true,'Sheet','Matrix_2')
    writetable(matrix_3_inner,filename,'WriteRowNames',true,'Sheet','Matrix_3_inner')
    writetable(matrix_3_outer,filename,'WriteRowNames',true,'Sheet','Matrix_3_outer')
else
    filename = "excel_files/" +  "trialData_mlp." + "xlsx";
    writetable(matrix_1_mlp,filename,'WriteRowNames',true,'Sheet','Matrix_1')
    writetable(matrix_1_pd_mlp,filename,'WriteRowNames',true,'Sheet','Matrix_1pd')
    writetable(matrix_2_mlp,filename,'WriteRowNames',true,'Sheet','Matrix_2')
    writetable(matrix_3_inner_mlp,filename,'WriteRowNames',true,'Sheet','Matrix_3_inner')
    writetable(matrix_3_outer_mlp,filename,'WriteRowNames',true,'Sheet','Matrix_3_outer')
end
