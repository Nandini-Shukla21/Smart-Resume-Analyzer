import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";
import { Upload, FileText, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedFileTypes?: string[];
  maxSize?: number; // in bytes
  className?: string;
  placeholder?: string;
}

export const FileUpload = ({
  onFileSelect,
  acceptedFileTypes = ['.pdf', '.docx', '.doc'],
  maxSize = 5 * 1024 * 1024, // 5MB default
  className,
  placeholder = "Upload your resume"
}: FileUploadProps) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string>("");

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc']
    },
    maxSize,
    multiple: false,
    onDrop: (acceptedFiles, rejectedFiles) => {
      setError("");
      
      if (rejectedFiles.length > 0) {
        const rejection = rejectedFiles[0];
        if (rejection.errors[0]?.code === 'file-too-large') {
          setError(`File is too large. Maximum size is ${maxSize / (1024 * 1024)}MB`);
        } else if (rejection.errors[0]?.code === 'file-invalid-type') {
          setError('Invalid file type. Please upload PDF or DOCX files only.');
        }
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setSelectedFile(file);
        onFileSelect(file);
      }
    }
  });

  const removeFile = () => {
    setSelectedFile(null);
    setError("");
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className={cn("w-full", className)}>
      <AnimatePresence mode="wait">
        {!selectedFile ? (
          <div
            className={cn(
              "glass-card border-2 border-dashed cursor-pointer transition-all duration-300",
              isDragActive ? "border-primary bg-primary/5 scale-[1.02]" : "border-glass-border hover:border-primary/50",
              "flex flex-col items-center justify-center p-8 text-center min-h-[200px]"
            )}
            {...getRootProps()}
          >
            <input {...getInputProps()} />
            
            <motion.div
              animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
              transition={{ duration: 0.2 }}
            >
              <Upload className={cn(
                "w-12 h-12 mb-4 transition-colors",
                isDragActive ? "text-primary" : "text-muted-foreground"
              )} />
            </motion.div>
            
            <h3 className="text-lg font-semibold mb-2">
              {isDragActive ? "Drop your resume here" : placeholder}
            </h3>
            
            <p className="text-muted-foreground mb-4">
              {isDragActive 
                ? "Release to upload" 
                : "Drag and drop your file here, or click to browse"
              }
            </p>
            
            <div className="text-sm text-muted-foreground">
              <p>Supported formats: {acceptedFileTypes.join(', ')}</p>
              <p>Maximum size: {formatFileSize(maxSize)}</p>
            </div>
          </div>
        ) : (
          <motion.div
            key="selected"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="glass-card border border-success/30 bg-success/5"
          >
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <FileText className="w-8 h-8 text-success" />
                <div>
                  <p className="font-medium text-foreground">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>
              
              <button
                onClick={removeFile}
                className="p-2 hover:bg-destructive/10 rounded-lg transition-colors"
              >
                <X className="w-4 h-4 text-destructive" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg"
        >
          <p className="text-sm text-destructive">{error}</p>
        </motion.div>
      )}
    </div>
  );
};